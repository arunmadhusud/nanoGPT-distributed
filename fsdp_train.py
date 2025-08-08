from model import GPT, GPTConfig
from utils import apply_sac, DataLoaderLite
import os
import math
import time
import torch
import wandb
import torch.nn.functional as F

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict, set_model_state_dict, set_optimizer_state_dict, StateDictOptions
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

      
#----------------------------------------------------------------------------------------------------------------------------------

# set up FSDP (Fully Sharded Data Parallel).
# torchrun command sets the env variables WORLD_SIZE
assert torch.cuda.is_available(), "CUDA is not available, please run on a multi-GPU machine"
torch.distributed.init_process_group(backend='nccl')
device_type = torch.accelerator.current_accelerator().type
ddp_world_size = int(os.environ['WORLD_SIZE'])
dp_mesh = init_device_mesh(device_type, (int(os.environ["WORLD_SIZE"]),))
ddp_rank = dp_mesh.get_rank()
ddp_local_rank = dp_mesh.get_local_rank()
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.

# set random seed for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 32768 # This is just for demo purpose. Karpathy's tutorial uses 2^^19 tokens per batch. 
B = 4 # batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# create a DataLoaderLite instance
train_loader = DataLoaderLite(B=B, T=T, master_process=master_process, process_rank=ddp_rank, num_processes=ddp_world_size)

'''
Set the precision to high for better training speed when using Ampere/Hopper GPUs
''' 
# torch.set_float32_matmul_precision('high')

# create a GPT model instance (model is loaded initially on CPU)
model = GPT(GPTConfig(vocab_size=50304)) 

# Save initialized checkpoint using the state dict APIs.
# We will reload this checkpoint later to the parallel model. You can read more about state dict APIs here:
# https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#state-dict-with-dcp-apis
model_state_dict = get_model_state_dict(
    model=model,
    options=StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
    )
)
if master_process:
    torch.save(model_state_dict, "model_state_dict.pt")
    print("model initial state dict saved")    
dist.barrier() # Synchronize all processes to ensure the model state dict is saved before proceeding
del model_state_dict

'''
FSDP2 Mixed Precision Policy enables us to cast float32 to bfloat16 forward/backward computation (faster training) and
Upcasting gradients to float32 for reduce-scatter to preserve accuracy. 
https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.MixedPrecisionPolicy
Uncomment the mp_policy lines below to enable mixed precision training in Ampere/Hopper GPUs. bfloat16 is not supported on older GPUs like T4.
'''
# mp_policy = MixedPrecisionPolicy(
#     param_dtype=torch.bfloat16,
#     reduce_dtype=torch.float32,
# )

'''
Let's Shard the model using FSDP2.
To achieve communication and computation overlap, the user must apply fully_shard to different modules (hence constructing multiple communication groups) plus the root module. 
For transformer architectures, this conventionally means applying fully_shard to each transformer block and then to the overall model.
https://github.com/pytorch/pytorch/issues/114299
'''
# Shard each transformer block individually first
for block in model.layers.values():
    fully_shard(block, 
                mesh=dp_mesh,
                reshard_after_forward=True, # reshards parameters after forward and all-gathers in backward.
                # mp_policy=mp_policy
                )

# Shard the entire model (this handles wte, wpe, ln_f, and lm_head together)
fully_shard(model, 
            mesh=dp_mesh,
            reshard_after_forward=False, # keeps the unsharded parameters in memory after forward and avoids the all-gather in backward. 
            # mp_policy=mp_policy
            )

# Reload the model state dict for re-initialization
model_state_dict = torch.load("model_state_dict.pt")
dist.barrier()

'''
We can load a full state dict into a FSDP2 model with set_model_state_dict. With broadcast_from_rank0=True, we can load the full state dict 
only on rank 0 to avoid peaking CPU memory and broadcast them to other ranks.
https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#state-dict-with-dcp-apis
Model parameters are loaded to GPU memory in a sharded manner now.
'''
set_model_state_dict(
    model=model,
    model_state_dict=model_state_dict,
    options=StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
    ),
)
dist.barrier()

'''
Set the learning rate schedule based on GPT-2 training.
'''
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50 
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# Compile the model after sharding
use_compile = True 
if use_compile:
    model = torch.compile(model)

# Apply Selective Activation Checkpointing to the model
set_sac = True
if set_sac:
    if master_process:
        print("Applying Selective Activation Checkpointing (SAC) to the model...")
    model = apply_sac(model)


# Enable wandb logging if master process
wandb_run = False
if master_process and wandb_run:
    wandb.login()
    wandb.init(
        project="nanoGPT-distributed",
        name=f"FSDP2-rank-{ddp_world_size}-gacc-{grad_accum_steps}",
        config={
            "total_batch_size_tokens": total_batch_size,
            "micro_batch_size": B,
            "sequence_length": T,
            "gradient_accumulation_steps": grad_accum_steps,
            "max_steps": max_steps,
            "max_lr": max_lr,
            "world_size": ddp_world_size,
            "selective_activation_checkpointing": True,
        },
    )
    
'''
Define the optimizer based on the GPT-2 training.
'''
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type, rank = ddp_rank)

'''
Training loop using FSDP2.
We use gradient accumulation to simulate a larger batch size. We accumulate gradients for grad_accum_steps and then step the optimizer.
Set model.set_requires_gradient_sync to True only on the last micro step to ensure that gradients are synchronized across all ranks. 
(https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_requires_gradient_sync)
'''
for step in range(max_steps):
    t0 = time.time() 
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if  (micro_step < grad_accum_steps - 1) :
            model.set_requires_gradient_sync(False)
        else: # micro_step == grad_accum_steps - 1
            model.set_requires_gradient_sync(True)            
        logits = model(x)
        # Compute loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()                  

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Dtensor, sharded on GPUs  
    '''
    loss_accum is updated on each rank. We need to average the loss_acum across all ranks to get the final loss value for logging.
    Since the gradient norm is a DTensor (stored as _NormPartial), to compute the total norm across all ranks, we need to aggregate the squared local norms and then take the square root of their sum, i.e., compute sqrt(a² + b²) where a and b are the local norms on different ranks. 
    This can be achieved by just calling .full_tensor(). 
    '''
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    full_norm = norm.full_tensor() # get the full tensor
  
    # determine and set the learning rate for this iterationprint(norm)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:        
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {full_norm.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        if wandb_run:
            wandb.log({
                "train_loss": loss_accum.item(),
                "learning_rate": lr,
                "grad_norm": full_norm.item(),
                "tokens_per_sec": tokens_per_sec
            }, step=step)


model_save = False
if model_save:
    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )
    optim_state_dict = get_optimizer_state_dict(
                model=model,
                optimizers=optimizer,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )
    # Only save on master process to avoid duplicate saves
    if master_process:
        torch.save(model_state_dict, "model_state_dict.pt")
        torch.save(optim_state_dict, "optim_state_dict.pt")
        print("model and optimizer state dicts saved")
    dist.barrier() # Synchronize all processes to ensure the model state dict is saved before proceeding
    
if master_process and wandb_run:
    wandb.finish()  # Finish the wandb run


# Clean up the distributed environment
torch.distributed.destroy_process_group()

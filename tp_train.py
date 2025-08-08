
from model import GPT, GPTConfig
from torch.nn import functional as F
from utils import apply_sac, DataLoaderLite, compare_model_weights
import os
import math
import time
import torch
import torch.nn as nn
import wandb


from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel,
    loss_parallel
)
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict, set_model_state_dict, set_optimizer_state_dict, StateDictOptions


# set up Tensor Parallelism.
# torchrun command sets the env variables WORLD_SIZE
assert torch.cuda.is_available(), "CUDA is not available, please run on a multi-GPU machine"
# torch.distributed.init_process_group(backend='nccl')
device_type = torch.accelerator.current_accelerator().type
tp_world_size = int(os.environ['WORLD_SIZE'])
tp_mesh = init_device_mesh(device_type, (int(os.environ["WORLD_SIZE"]),))
tp_rank = tp_mesh.get_rank()
tp_local_rank = tp_mesh.get_local_rank()
device = f'cuda:{tp_local_rank}'
master_process = tp_rank == 0

# set random seed for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 32768 # This is just for demo purpose. Karpathy's tutorial uses 2^^19 tokens per batch. 
B = 4 # batch size
T = 1024 # sequence length
assert total_batch_size % (B * T ) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# create a DataLoaderLite instance
train_loader = DataLoaderLite(B=B, T=T, master_process=master_process)

'''
Set the precision to high for better training speed when using Ampere/Hopper GPUs
''' 
# torch.set_float32_matmul_precision('high')

# create a GPT model instance (model is loaded initially on CPU)
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2

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
Now we will parallelize the model using Tensor Parallelism + Sequence Parallelism.
We will shard the transformer blocks and remaining layers (embeddings, layernorm, lm_head) separately.
Parallelization plan should be defined in away to maximize efficiency and minimize communication overhead.
please refer to the documentation for more details on how to parallelize a Transformer (Llama) model:
https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html
In comparison to the Llama model, for GPT-2, additionally we have to shard positional embeddings, 
for which we use RowwiseParallel with inputs replicated and outputs sharded (at sequence dimension, dim=0).
and for attention blocks, we will only have one input (no rotary embeddings).
'''

model = parallelize_module(
    model,
    tp_mesh,
    {
        "wte": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "wpe": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(0),
        ),
        "ln_f": SequenceParallel(),
        "lm_head": ColwiseParallel(
            input_layouts=Shard(1),
            use_local_output=False, # keep the output sharded and output stay as a DTensor to work with the loss_parallel.
        ),
    }
)

for layer_id in range(len(model.layers)):
    layer_tp_plan = {
        "ln_1": SequenceParallel(),
        "attn": PrepareModuleInput(
            input_layouts=(Shard(1),),           # Only one input
            desired_input_layouts=(Replicate(),), # Only one input
        ),
        "attn.c_attn": ColwiseParallel(use_local_output=False),
        "attn.c_proj": RowwiseParallel(output_layouts=Shard(1)),
        "ln_2": SequenceParallel(),
        "mlp": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "mlp.c_fc": ColwiseParallel(),
        "mlp.c_proj": RowwiseParallel(output_layouts=Shard(1)),
    }

    parallelize_module(
        module=model.layers[str(layer_id)],
        device_mesh=tp_mesh,
        parallelize_plan=layer_tp_plan
    )

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

#--------------------------------------------------------------------------------

# Enable wandb logging if master process
wandb_run = False
if master_process and wandb_run:
    wandb.login()
    wandb.init(
        project="nanoGPT-distributed",
        name=f"TP+SP-rank-{tp_world_size}-gacc-{grad_accum_steps}",
        config={
            "total_batch_size_tokens": total_batch_size,
            "micro_batch_size": B,
            "sequence_length": T,
            "gradient_accumulation_steps": grad_accum_steps,
            "max_steps": max_steps,
            "max_lr": max_lr,
            "world_size": tp_world_size,
            "selective_activation_checkpointing": True,
        },
    )


# Reload the model state dict for re-initialization
model_state_dict = torch.load("model_state_dict.pt")
dist.barrier()
'''
Reload the model state dict from CPU to the parallelized model on GPUs.
You can read more about the state dict APIs here:
https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#state-dict-with-dcp-apis
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
del model_state_dict


'''
For our setup, we need to ensure that the weight sharing is working correctly.
So define weight tying for the input and output embeddings again.
'''
# model.transformer.wte.weight = model.lm_head.weight  # Weight tying

# # Compare the loaded model weights with the current model weights (for verification)
# model_state_dict = torch.load("model_state_dict.pt")
# dist.barrier()
# compare_model_weights(
#     model_state_dict=model_state_dict,
#     model=model,
#     rank=tp_rank,
#     device=device
# )

# Compile the model after parallelization
use_compile = True 
if use_compile:
    def apply_compile(model: nn.Module):
        for layer_id in range(len(model.layers)):
            transformer_block = torch.compile(model.layers[str(layer_id)], fullgraph=True)
            model.layers[str(layer_id)] = transformer_block

        print(f"Applied torch.compile to {len(model.layers)} transformer blocks")
        return model
    model = apply_compile(model)

# Apply Selective Activation Checkpointing to the model
set_sac = True
if set_sac:
    if master_process:
        print("Applying Selective Activation Checkpointing (SAC) to the model...")
    # Initialize SAC
    sac = SAC()
    # Apply SAC to the model
    model = sac.apply_sac(model)

'''
Define the optimizer based on the GPT-2 training.
'''
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type, rank = tp_rank)

'''
Training loop
We use gradient accumulation to simulate a larger batch size. We accumulate gradients for grad_accum_steps and then step the optimizer.
We also use loss_parallel to efficiently compute the cross-entropy loss when the model outputs are sharded on the (often huge) vocabulary dimension, without gathering all the model outputs to every single GPU.
https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html#apply-loss-parallel
'''
for step in range(max_steps):
    t0 = time.time() 
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, _ = model(x) 
        with loss_parallel():
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()                    

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
    # norm is a Dtensor, values are stored on each GPU as Partial("sum"). Call full_tensor() to get the full norm across all GPUs.
    full_norm = norm.full_tensor()
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    if tp_rank== 0:
        # loss_accum is replicated across all GPUs, so we can just call item() to get the value for logging
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {full_norm.item():.6f}| dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
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
        print("TP checkpoint saved!")
    
if tp_rank == 0 and wandb_run:
    wandb.finish()

# Clean up the distributed environment
torch.distributed.destroy_process_group()
    

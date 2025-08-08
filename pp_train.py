import os
import math
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
from model import GPT, GPTConfig
from utils import apply_sac, DataLoaderLite
import wandb

#---------------------------------------------------------------------------------------------------

import os
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict, set_model_state_dict, set_optimizer_state_dict, StateDictOptions


# set up Tensor Parallelism.
assert torch.cuda.is_available(), "Pipeline parallelism requires CUDA"
device_type = torch.accelerator.current_accelerator().type
# torchrun command sets the env variables WORLD_SIZE
pp_mesh = init_device_mesh(device_type, (int(os.environ["WORLD_SIZE"]),))
# torch.distributed.init_process_group(backend='nccl')
pp_rank = pp_mesh.get_rank()
pp_local_rank = pp_mesh.get_local_rank()
pp_world_size = pp_mesh.size()
device = f'cuda:{pp_local_rank}'
torch.cuda.set_device(device)


# Pipeline specific settings
num_stages = pp_world_size # Number of pipeline stages, same as number of GPUs in this case
stage_index = pp_rank # Current stage index
master_process = stage_index == num_stages - 1 # Last stage will be the master process

# set random seed for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 32768 # This is just for demo purpose. Karpathy's tutorial uses 2^^19 tokens per batch.
B = 4 # batch size
T = 1024 # sequence length
assert total_batch_size % (B * T ) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
if stage_index == 0:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")  

"""
In simple pipeline parallelism, during forward pass and backward pass, each GPU has to wait for the previous stage to finish processing a batch. This can lead to GPU idle time (bubbles).
To mitigate this, we can use microbatches. We will split a single batch into microbatches, and each stage will process one microbatch at a time. This allows us to overlap the processing of different microbatches across stages.
and reduce bubble time. The ratio of bubble time to processing time is called the bubble ratio calculated as:
bubble_ratio = (num_stages - 1) / n_microbatches.
"""
n_microbatches = 4 # Number of microbatches per stage. In our case, each micobatch will consist of 1 data sample (B/n_microbatches)

if master_process:
    print(f"Pipeline parallelism with {num_stages} stages")
    print(f"Batch size (B): {B}")
    print(f"Number of microbatches: {n_microbatches}")

train_loader = DataLoaderLite(B=B, T=T, master_process=master_process)

# Create model
model = GPT(GPTConfig(vocab_size=50304))

# def get_modules(model, num_stages):
#     """
#     Get the modules for each stage in the pipeline.
#     We will partition the model into num_stages parts, each containing a subset of the transformer layers.
#     """
#     # Calculate the number of transformer layers per stage
#     layers_per_stage = model.config.n_layer // num_stages
#     module_names_per_stage = []
    
#     for stage in range(num_stages):
#         stage_modules = []        
#         if stage == 0:
#             stage_modules.extend(['transformer.wte', 'transformer.wpe'])
#             start_layer = 0
#             end_layer = layers_per_stage
#         elif stage == num_stages - 1:
#             start_layer = stage * layers_per_stage
#             end_layer = model.config.n_layer
#             for layer_idx in range(start_layer, end_layer):
#                 stage_modules.append(f'transformer.h.{layer_idx}')
#             stage_modules.extend(['transformer.ln_f', 'lm_head'])
#         else:
#             start_layer = stage * layers_per_stage
#             end_layer = (stage + 1) * layers_per_stage
        
#         # Add transformer layers for non-last stages or if not handled above
#         if stage != num_stages - 1:
#             for layer_idx in range(start_layer, end_layer):
#                 stage_modules.append(f'transformer.h.{layer_idx}')
        
#         module_names_per_stage.append(stage_modules)        
    
#     # if master_process:
#     #     print(f"Layers per stage: {layers_per_stage}")
#     #     print(f"Module names per model part:")
#     #     for i, stage_modules in enumerate(module_names_per_stage):
#     #         print(f"  Stage {i}: {stage_modules}")
    
#     return module_names_per_stage

# # Get the modules for each stage
# module_names_per_stage = get_modules(model, num_stages)

# # https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html#step-1-partition-the-transformer-model
# def build_stage_from_modules(
#         stage_idx: int, module_names: list[str], num_stages: int, whole_model: nn.Module
#     ) :
#     """
#     Partition the model based on the provided module names for the given stage index.
#     The transformer blocks which are not part of the current stage will be removed.
#     The forward function of our model requires (token embeddings, position embeddings, layer norms, lm_head), so we will keep them as None if they are not part of the current stage.
#     The function returns a PipelineStage object for the current stage and the modified model.
#     """
#     model = copy.deepcopy(whole_model) # Create a copy of the model 
#     modules_to_keep = set(module_names)
    
#     transformer = model.transformer    
#     for component_name in list(transformer.keys()):
#         component = transformer[component_name]        
#         full_component_name = f"transformer.{component_name}"  

#         if isinstance(component, nn.ModuleList):
#             layers_to_keep = []  

#             for i, layer in enumerate(component):
#                 layer_name = f"{full_component_name}.{i}"
#                 if layer_name in modules_to_keep:
#                     layers_to_keep.append(i)     

#             if layers_to_keep:
#                 new_layers = nn.ModuleList()
#                 for i in layers_to_keep:
#                     new_layers.append(component[i])
#                 transformer[component_name] = new_layers
#             else:
#                 del transformer[component_name] # Transformer blocks not in this stage will be removed
                
#         elif full_component_name in modules_to_keep:
#             # Keep components like wte, wpe, ln_f as they are
#             pass
#         else:
#             # Remove components not in modules_to_keep
#             transformer[component_name] = None # wte, wpe, ln_f will be set to None if not in this stage

    
#     # Handle lm_head (outside transformer)
#     if 'lm_head' not in modules_to_keep:
#         if hasattr(model, 'lm_head'):
#             model.lm_head = None # lm_head will be set to None if not in this stage

#     stage = PipelineStage(
#         model,
#         stage_idx,
#         num_stages,
#         device,
#     )
#     return stage, model



    
# Build the stage for the current stage index
# This will create a PipelineStage object with the model partitioned for the current stage
# The model will only contain the modules that are part of the current stage
# stage, model = build_stage_from_modules(stage_index, module_names_per_stage[stage_index], num_stages, model)

def partition_model_for_stage(model, stage_idx, num_stages):
    model = copy.deepcopy(model)  # Create a copy first!

    layers_per_stage = model.config.n_layer // num_stages
    
    if stage_idx == 0:
        # First stage: keep embeddings + first set of layers
        start_layer = 0
        end_layer = layers_per_stage
        # Remove later layers
        for i in range(end_layer, model.config.n_layer):
            del model.layers[str(i)]
        # Remove final components
        model.ln_f = None
        model.lm_head = None
        
    elif stage_idx == num_stages - 1:
        # Last stage: keep final layers + output components
        start_layer = stage_idx * layers_per_stage
        # Remove embeddings
        model.wte = None
        model.wpe = None
        # Remove earlier layers
        for i in range(start_layer):
            del model.layers[str(i)]
            
    else:
        # Middle stage: keep only assigned layers
        start_layer = stage_idx * layers_per_stage
        end_layer = (stage_idx + 1) * layers_per_stage
        # Remove embeddings and final components
        model.wte = None
        model.wpe = None
        model.ln_f = None
        model.lm_head = None
        # Remove layers outside this stage's range
        for i in range(model.config.n_layer):
            if i < start_layer or i >= end_layer:
                del model.layers[str(i)]
            
    stage = PipelineStage(
        model,
        stage_idx,
        num_stages,
        device,
    )
    return stage, model

stage, model = partition_model_for_stage(model, stage_index, num_stages)

model = model.to(device)

# Compile the model after parallelization
use_compile = True 
if use_compile:
    model = torch.compile(model)

# Apply Selective Activation Checkpointing to the model
set_sac = True
if set_sac:
    if master_process:
        print("Applying Selective Activation Checkpointing (SAC) to the model...")
    model = apply_sac(model)

if master_process:
    print(f"Pipeline created with {num_stages} stages")
    print(f"Current stage {stage_index} on device {device}")

# Learning rate schedule
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

# Enable wandb logging if master process
wandb_run = False
if master_process and wandb_run:
    wandb.login()
    wandb.init(
        project="nanoGPT-distributed",
        name=f"PP-{pp_world_size}-gacc-{grad_accum_steps}",
        config={
            "total_batch_size_tokens": total_batch_size,
            "micro_batch_size": B,
            "sequence_length": T,
            "gradient_accumulation_steps": grad_accum_steps,
            "max_steps": max_steps,
            "max_lr": max_lr,
            "world_size": pp_world_size,
            "selective_activation_checkpointing": True,
        },
    )

# Setup optimizer 
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type="cuda", rank=0) # Setting rank=0 to print number of parameters for all GPUs


# Define loss function for pipeline
def loss_fn(outputs, targets):
    return F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1)) / ( grad_accum_steps * n_microbatches )


# Create pipeline schedule
# We will use GPipe for pipeline parallelism.
schedule = ScheduleGPipe(stage, n_microbatches=n_microbatches, loss_fn=loss_fn)

@torch.no_grad()
def clip_grad_norm_(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool = True,
    pp_mesh = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.
    Gradient norm clipping requires computing the gradient norm over the entire model.`torch.nn.utils.clip_grad_norm_` does this for TP and DP, but not PP.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    norm_type = 2.0
    total_norm = torch.nn.utils.get_total_norm(grads, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite, foreach=foreach)
    total_norm **= norm_type
    dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
    total_norm **= 1.0 / norm_type
    torch.nn.utils.clip_grads_with_norm_(model.parameters(), max_norm=max_norm, total_norm=total_norm, foreach=foreach)
    return total_norm

# Training loop
for step in range(max_steps):
    t0 = time.time() 
    
    # Pipeline parallelism training (no gradient accumulation)
    optimizer.zero_grad()   
    
    # Reset loss accumulation for the last stage
    if stage_index == pp_world_size-1: 
        loss_accum = 0

    # Different behavior for different stages
    for micro_step in range(grad_accum_steps):
        # Get one batch and let pipeline handle microbatching
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if stage_index == 0:
            # First stage receives inputs and provides them to pipeline
            schedule.step(x)
        elif stage_index == pp_world_size-1:
            # Other stages just participate in the pipeline
            losses = []
            schedule.step(target=y, losses=losses)
            # print(f"losses : {losses}")
            loss_accum += (sum(losses))
            
        else:            
            schedule.step()

    # Clip gradients
    total_norm = clip_grad_norm_(model, max_norm=1.0, norm_type=2.0, error_if_nonfinite=True, foreach=True, pp_mesh=pp_mesh)

    # Set learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    
    # Calculate tokens processed (effective batch size)
    tokens_processed = B * T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    
    if stage_index == pp_world_size-1:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {total_norm.item():.6f}|dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        if wandb_run:
            wandb.log({
                "train_loss": loss_accum.item(),
                "learning_rate": lr,
                "grad_norm": total_norm.item(),
                "tokens_per_sec": tokens_per_sec
            }, step=step)


if master_process and wandb_run:
    wandb.finish()  # Finish the wandb run

model_save = False
if model_save:
    # Save model weights
    torch.save(model.state_dict(), f"model_stage_{stage_index}_model.pth")
    
    # Save optimizer state
    torch.save(optimizer.state_dict(), f"model_stage_{stage_index}_optim.pth")


# Clean up the distributed environment
torch.distributed.destroy_process_group()
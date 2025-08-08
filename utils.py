import torch
import torch.nn as nn
import functools
import torch.distributed as dist
import math
import tiktoken
from typing import Optional
import os
import requests


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts
)

def apply_sac(model: nn.Module):
    """Apply activation checkpointing to the model."""
    aten = torch.ops.aten
    compute_intensive_ops = [
        aten.mm,                              # All your Linear layers
        aten.addmm,                           # Alternative MM operation
        aten.scaled_dot_product_attention,    # attention computation
        aten._scaled_dot_product_flash_attention,  # Flash attention variant
        aten._scaled_dot_product_efficient_attention,  # Efficient attention variant
    ]
    
    def policy_fn(ctx, op, *args, **kwargs):
        """Simple policy: save compute-intensive ops, recompute everything else"""
        if op in compute_intensive_ops:
            return CheckpointPolicy.MUST_SAVE
        else:
            return CheckpointPolicy.PREFER_RECOMPUTE
            
    def apply_ac_to_transformer_block(module: nn.Module):
        return ptd_checkpoint_wrapper(
            module,
            context_fn=functools.partial(create_selective_checkpoint_contexts, policy_fn),
            preserve_rng_state=False,
        )
    
    for layer_id, layers in model.layers.items():
        transformer_block = apply_ac_to_transformer_block(layers)
        model.layers[layer_id] = transformer_block

    return model

class DataLoaderLite:
    def __init__ (self, B, T, master_process: bool = False, process_rank: Optional[int] = 0, num_processes: Optional[int] = 1):
        """
        A simple data loader that loads a text file and returns batches of tokens.
        Args:
            B (int): Batch size.
            T (int): Sequence length.
            master_process (bool): If True, download the dataset if not already present.
            process_rank (int): Rank of the current process. We will use this in DataParallel only. 
            num_processes (int, optional): Total number of processes. We will use this in DataParallel only. Defaults to 1.
        """
        # download the tiny shakespeare dataset if not already present
        if master_process:
            input_file_path = 'input.txt'
            if not os.path.exists(input_file_path):
                data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
                with open(input_file_path, 'w', encoding='utf-8') as f:
                    f.write(requests.get(data_url).text)
        # synchronize all processes to ensure the file is downloaded
        dist.barrier()
        
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        enc = tiktoken.get_encoding('gpt2')
        with open('input.txt', 'r') as f:
            text = f.read()
        self.tokens = enc.encode(text)
        if master_process:
            print(f"loaded {len(self.tokens)} tokens")
        
        #current state
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = torch.tensor(self.tokens[self.current_position : self.current_position+B*T+1])
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = B * T * self.process_rank
        return x, y

def compare_model_weights(model_state_dict, model, rank, device):
    """
    Compare loaded model state dict with current model weights.
    
    Args:
        model_state_dict: Dictionary from torch.load("model_state_dict.pt")
        model: Current model instance
        rank: Process rank (only rank 0 will print)
        device: Device to move tensors to for comparison
    
    Returns:
        bool: True if all weights match, False otherwise
    """

    # Get the current weights from B
    current_state_dict = model.state_dict()
 
    mismatch_found = False
    total_params = 0
    different_params = 0
    
    for key in model_state_dict :
        if key not in current_state_dict:
            print(f"Key '{key}' not in current model.")
            mismatch_found = True
            continue
        
        loaded_param = model_state_dict[key].to(device)
        current_param = current_state_dict[key].full_tensor()
        
        same = torch.allclose(loaded_param, current_param, atol=1e-6)
        total_params += loaded_param.numel()
        
        if not same:
            # Calculate difference statistics
            diff = torch.abs(loaded_param - current_param)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            if rank==0 :
                print(f"Mismatch in: {key}")
                print(f"Max difference: {max_diff:.8f}")
                print(f"Mean difference: {mean_diff:.8f}")
                print(f"Loaded stats: mean={loaded_param.mean().item():.6f}, std={loaded_param.std().item():.6f}")
                print(f"Fresh stats:  mean={current_param.mean().item():.6f}, std={current_param.std().item():.6f}")

            different_params += loaded_param.numel()
            mismatch_found = True
    
    if not mismatch_found and rank==0:
        print("All weights match perfectly!")
    else:
        if rank==0:
            print(f"Summary:")
            print(f"Total parameters: {total_params:,}")
            print(f"Different parameters: {different_params:,}")
            print(f"Percentage different: {100 * different_params / total_params:.2f}%")
    






        

# ==========================================================
# Script for DUS (Depth Up-Scaling)
# Author: ShinoharaHare
# Inspired by the SOLAR project
# ==========================================================

import copy
import re

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(
    model_path: str,
    output_path: str,
    num_merged_layers: int = 8
):
    """
    Perform depth up-scaling (DUS) on a pretrained LLM.

    Args:
        model_path (str): Path to the original pretrained model.
        output_path (str): Path where the scaled model will be saved.
        num_merged_layers (int): Number of layers to merge/restructure.
                                 Controls how much scaling is applied.
    """

    # Load tokenizer and model from the given path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype='auto',          # automatically choose optimal precision
        low_cpu_mem_usage=True       # reduce memory usage while loading
    )

    # Original number of transformer layers
    num_layers = model.config.num_hidden_layers

    # Compute the number of layers after depth up-scaling
    # Formula: double the remaining layers after merging
    num_scaled_layers = 2 * (num_layers - num_merged_layers)

    # Copy model configuration and update with new layer count
    scaled_config = copy.copy(model.config)
    scaled_config.num_hidden_layers = num_scaled_layers

    # Initialize an empty model on 'meta' device (saves memory)
    with torch.device('meta'):
        scaled_model = AutoModelForCausalLM.from_config(scaled_config)

    # Fetch original model weights
    state_dict = model.state_dict()
    scaled_state_dict = {}

    # Regex pattern to identify layer indices in parameter keys
    layer_pattern = re.compile(r'layers.(\d+).')

    # Remap weights from original model into the scaled model
    for k in scaled_model.state_dict().keys():
        if (
            (m := layer_pattern.search(k)) and
            (layer_idx := int(m.group(1))) >= num_layers - num_merged_layers
        ):
            # If layer index is in the merged region, shift it
            target_layer_idx = layer_idx - num_layers + 2 * num_merged_layers
            target_key = layer_pattern.sub(f'layers.{target_layer_idx}.', k)
        else:
            # Otherwise, keep the same key
            target_key = k

        print(f'{k} ← {target_key}')  # log mapping
        scaled_state_dict[k] = state_dict[target_key].clone()

    # Load remapped weights into the scaled model
    scaled_model.load_state_dict(scaled_state_dict, strict=True, assign=True)

    # Save the new scaled model and tokenizer
    scaled_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    # Fire CLI allows running the script with arguments directly from terminal
    fire.Fire(main)

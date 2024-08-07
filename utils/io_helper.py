
def print_grad_status(module, module_path="", file_path="grad_status.txt"):
    with open(file_path, 'w') as file:
        for name, param in module.named_parameters():
            print(f"{module_path + name} -> requires_grad={param.requires_grad}", file=file)


import torch
import json
def load_stats(filename):
    # Read the JSON file
    with open(filename, 'r') as f:
        stats = json.load(f)
    
    # Convert lists back to tensors
    mean = torch.tensor(stats['mean'])
    std = torch.tensor(stats['std'])
    
    return mean, std
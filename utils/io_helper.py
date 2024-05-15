
def print_grad_status(module, module_path="", file_path="grad_status.txt"):
    with open(file_path, 'w') as file:
        for name, param in module.named_parameters():
            print(f"{module_path + name} -> requires_grad={param.requires_grad}", file=file)
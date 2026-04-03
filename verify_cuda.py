import torch

assert torch.cuda.is_available(), "CUDA is NOT available — check drivers/PyTorch install"

print(f"CUDA available : True")
print(f"Device count   : {torch.cuda.device_count()}")
print(f"Device name    : {torch.cuda.get_device_name(0)}")
print(f"CUDA version   : {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")

device = torch.device("cuda")
x = torch.tensor([1.0, 2.0, 3.0], device=device)
print(f"Tensor on {x.device}: {x}")
print("Verification passed.")

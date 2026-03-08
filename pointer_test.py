import torch

x = torch.tensor([0.0, 1,0, 2.0]).to(device="cuda")

print(x.data_ptr)
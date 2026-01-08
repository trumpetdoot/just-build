import torch
import numpy as np
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)
x_ones = torch.ones_like(x_data)
print(x_ones)
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand)

shape = (2, 3, 3)
rand_tensor = torch.rand(shape)
one_tensor = torch.ones(shape)
zero_tensor = torch.zeros(shape)

print(rand_tensor)
print(one_tensor)
print(zero_tensor)


# Tensor Attributes
tensor = torch.rand(3, 4)

print(tensor.shape)
print(tensor.dtype)
print(tensor.device)

# Accelerator (For parallel processing)

print(torch.accelerator.current_accelerator()) # MPS for Apple

# Move tensor to accelerator
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())


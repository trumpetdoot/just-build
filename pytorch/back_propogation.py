# Back propogation is most popular algorithm for training nn. 
# Weights are adjusted based on gradient of loss function

import torch

x = torch.ones(5) # Input tensor
y = torch.zeros(3) # Expected tensor
w = torch.randn(5, 3, requires_grad=True) # Parameter 1
b = torch.randn(3, requires_grad=True) # Parameter 2
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) # loss function

# We aim to optimize parameters w and b 
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# Gradient calculation
loss.backward()
print(w.grad)
print(b.grad)


# In the case where we only want to apply a model (we already trained it) we can turn grad tracking off
# Also to mark parameters as frozen parameter, speed up computation since tracking tanks efficiency
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b

print(z.requires_grad)
# or 
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

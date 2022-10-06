import torch
import torch.nn as nn
loss = nn.L1Loss()
input = torch.randn(3, 5, requires_grad=True)
print (input)
target = torch.randn(3, 5)
output = loss(input, target)
print(output)
print (target)
loss.zero_grad()
output.backward()

print (input.grad)

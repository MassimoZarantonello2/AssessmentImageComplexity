import torch

values = torch.linspace(0.3, 2, 5)
values = values.tolist()
t = 'saturation'

x = [t] + values

print(x)
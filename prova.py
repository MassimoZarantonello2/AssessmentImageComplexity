import torch
import numpy as np

values = np.linspace(-128, 128, 5)
print(values)
values = torch.linspace(0.3, 2, 5)
print(values)
values = torch.linspace(0.3, 2, 5)
print(values)		
values = [-0.42, -0.19, 0.15, 0.32, 0.49]
print(values)		
values = torch.linspace(0.01, 0.1, 5)
print(values)		
values = torch.linspace(7, 1, 5, dtype=int)
print(values)
values = torch.linspace(3, 27, 5, dtype=int)
print(values)
values = torch.linspace(90, 10, 5, dtype=int)
print(values)
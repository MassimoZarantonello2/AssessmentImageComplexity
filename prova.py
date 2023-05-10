import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.models import inception_v3,Inception_V3_Weights
from PIL import Image
import os

model = models.inception_v3(weights = Inception_V3_Weights)

for child in model.children():
    print(child)

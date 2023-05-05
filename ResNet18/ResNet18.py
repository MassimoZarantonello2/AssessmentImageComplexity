import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os

if __name__ == '__main__':
    resnet18_GT = []
    i=1
    model = models.resnet18(weights = ResNet18_Weights.DEFAULT)
    fourth_layer = nn.Sequential(*list(model.children())[0:5])
    folder_path = './IC9600/images/'
    #resize the image to 224x224 and convert it to a torch tensor
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    with open ('./ResNet18/ResNet18_GT.txt', 'w') as f:
        image_list = os.listdir(folder_path)
        for image_name in image_list:
            image = Image.open(folder_path + image_name)
            image_name = image_name.replace(' ','_')
            input_tensor = preprocess(image)
            if(input_tensor.shape[0] == 1):
                input_tensor = input_tensor.repeat(3,1,1)
            input_batch = input_tensor.unsqueeze(0)
            output = fourth_layer(input_batch)
            complexity = output.mean().data.numpy()
            f.write(image_name + ' ' + str(complexity) + '\n')
            i+=1
            print(f"Processing image {i}/{len(image_list)}...", end="\r")
            
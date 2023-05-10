import torch.nn as nn
import torch
import torchvision.models as models
import torchvision.models as VGG16_weights
from torchvision import transforms
from PIL import Image
import os
import math

model = models.vgg16(pretrained=True)
n_layer =  nn.Sequential(*list(model.children())[:3])

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        #std=[0.229, 0.224, 0.225])
])

def get_complexity_score(input_tensor):
    img_tensor = input_tensor.unsqueeze(0)
    output = n_layer(img_tensor)
    return output.mean().data.numpy()

def compute_VGG16_IC9600():
    read_path = './IC9600/images/'
    save_path = './VGG-16/test_results/new_IC9600_VGG-16_fourth_layer_normalized.txt'
    i=1
    with open (save_path, 'w') as f:
        image_list = os.listdir(read_path)
        for image_name in image_list:
            i+=1
            print(f"Processing image {i}/{len(image_list)}...", end="\r")
            image = Image.open(read_path + image_name)
            image_name = image_name.replace(' ','_')
            if(image.mode != 'RGB'):
                image = image.convert('RGB')
            image = preprocess(image)
            complexity = get_complexity_score(image)
            f.write(image_name + ' ' + str(complexity) + '\n')
            
def compute_VGG16_Savoias():
    main_read_path = './Savoias-Dataset-master/images/'
    write_path = './VGG-16/test_results/new_Savoias_VGG16_fourth_layer_normalized.txt'
    categories_list = os.listdir(main_read_path)
    i=1
    with open(write_path, 'w') as f:
        for category in categories_list:
            read_path = main_read_path + category + '/'
            for image_name in os.listdir(read_path):
                print(f"Processing image {i}...", end="\r")
                i+=1      
                image = Image.open(read_path + image_name)
                output_image_name = category + '_' + image_name
                if(image.mode != 'RGB'):
                    image = image.convert('RGB')
                input_tensor = preprocess(image)
                complexity_score = get_complexity_score(input_tensor)
                f.write(output_image_name + ' ' + str(complexity_score) + '\n')
                
if __name__ == '__main__':
    '''IC9600'''
    compute_VGG16_IC9600()

    '''Savoias'''
    #compute_VGG16_Savoias()

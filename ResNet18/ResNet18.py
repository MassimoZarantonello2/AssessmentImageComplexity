import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os

model = models.resnet18(weights = ResNet18_Weights.DEFAULT)
fourth_layer = nn.Sequential(*list(model.children())[:5])
#resize the image to 224x224 and convert it to a torch tensor
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        #std=[0.229, 0.224, 0.225])
])

def get_complexity_score(input_tensor):
    input_batch = input_tensor.unsqueeze(0)
    output = fourth_layer(input_batch)
    complexity = output.mean().data.numpy()
    return complexity

def compute_ResNet18_IC9600():
    read_path = './IC9600/images/'
    save_path = './ResNet18/test_results/new_IC9600_ResNet18_fourth_layer_normalized.txt'
    i=0
    with open (save_path, 'w') as f:
        image_list = os.listdir(read_path)
        for image_name in image_list:
            i+=1
            print(f"Processing image {i}/{len(image_list)}...", end="\r")
            image = Image.open(read_path + image_name)
            image_name = image_name.replace(' ','_')
            if(image.mode != 'RGB'):
                image = image.convert('RGB')
            input_tensor = preprocess(image)
            complexity = get_complexity_score(input_tensor)
            f.write(image_name + ' ' + str(complexity) + '\n')
            
def compute_ResNet18_Savoias():
    main_read_path = './Savoias-Dataset-master/images/'
    write_path = './ResNet18/test_reuslts/Savoias_ResNet18_fourth_layer_normalized.txt'
    categories_list = os.listdir(main_read_path)
    i=0
    with open(write_path, 'w') as f:
        for category in categories_list:
            read_path = main_read_path + category + '/'
            for image_name in os.listdir(read_path):
                print(f"Processing image {i}...", end="\r")
                i+=1      
                image = Image.open(read_path + image_name)
                output_image_name = category.replace(' ','_') + '_' + image_name
                if(image.mode != 'RGB'):
                    image = image.convert('RGB')
                input_tensor = preprocess(image)
                complexity_score = get_complexity_score(input_tensor)
                f.write(output_image_name + ' ' + str(complexity_score) + '\n')
            
if __name__ == '__main__':   
    '''IC9600    '''
    compute_ResNet18_IC9600()

  
    '''Savoias   
    compute_ResNet18_Savoias()
'''
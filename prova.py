import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import torch
def blur(im,value):
    im = TF.gaussian_blur(T.ToTensor()(im), kernel_size=value)
    im = Image.fromarray(np.uint8(im.numpy().transpose(1, 2, 0) * 255))
    return im

def brightness(im,value):
    im_ycbcr = np.asarray(im.convert('YCbCr'))
    im_ycbcr_ = im_ycbcr.copy().astype(np.float32)
    im_ycbcr_[:,:,0] = np.clip(im_ycbcr_[:,:,0] + value, 0, 255)
    im_ycbcr_ = Image.fromarray(im_ycbcr_.astype(np.uint8), mode='YCbCr').convert('RGB')
    return im_ycbcr_

def contrast(im, value):
    im = TF.adjust_contrast(T.ToTensor()(im), contrast_factor=value)
    im = Image.fromarray(np.uint8(im.numpy().transpose(1, 2, 0) * 255))
    return im

def saturation(im, value):
    im = TF.adjust_saturation(T.ToTensor()(im), saturation_factor=value)
    im = Image.fromarray(np.uint8(im.numpy().transpose(1, 2, 0) * 255))
    return im

def hue(im, value):
    im = TF.adjust_hue(T.ToTensor()(im), hue_factor=value)
    im = Image.fromarray(np.uint8(im.numpy().transpose(1, 2, 0) * 255))
    return im

def jpeg(im, value):
    im.save('tmp.jpg', quality=value)
    im = T.ToTensor()(Image.open('tmp.jpg'))
    return im

def noise(im, value):
    im = T.ToTensor()(im)
    im = im + torch.randn(im.size()[1:])[None] * value
    im = Image.fromarray(np.uint8(im.numpy().transpose(1, 2, 0) * 255))
    return im

def posterize(im,value):
    im = TF.posterize(T.ToTensor()(im), bits=value)
    im = Image.fromarray(np.uint8(im.numpy().transpose(1, 2, 0) * 255))
    return im

    
def hflip(im,value):
    im = TF.hflip(T.ToTensor()(im))
    im = Image.fromarray(np.uint8(im.numpy().transpose(1, 2, 0) * 255))
    return im

blur_values = [3,9,15,21,27]
contrast_values = np.linspace(0.3, 2, 5)
brightness_values = np.linspace(-128, 128, 5)
saturation_values = np.linspace(0.3, 2, 5)
hue_values = [-0.42, -0.19, 0.15, 0.32, 0.49]
noise_values = np.linspace(0.01, 0.1, 5)
jpeg_values = torch.linspace(90, 10, 5, dtype=int)
posterize_values = torch.linspace(7, 1, 5, dtype=int)
hflip_values = [0, 1]

all_values = [blur_values,contrast_values, brightness_values, ]

functions = [blur, contrast, brightness,]
    
norm_im = Image.open("./IC9600/images/objects_bird_COCO_train2014_000000255051.jpg")
im = Image.open("./IC9600/images/objects_bird_COCO_train2014_000000255051.jpg")

x = None

for function,value in zip(functions,all_values):
    print(function.__name__)
    for v in value:
        print('     '+str(v))
        if x is None:
            x = function(norm_im,v)
        else:
            im = function(norm_im,v)
            x = np.concatenate((x,im),axis=1)
            
    x = np.concatenate((np.asarray(norm_im),x),axis=1)
    Image.fromarray(x).save("./images/"+function.__name__+".jpg")
    x = None

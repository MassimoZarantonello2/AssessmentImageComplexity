import torchvision
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image

def compute_blur(image,kernel_size=3):       #https://pytorch.org/vision/main/generated/torchvision.transforms.functional.gaussian_blur.html
    blur_adjusted_img = torchvision.transforms.functional.gaussian_blur(image,kernel_size)
    return blur_adjusted_img

def compute_adjusted_brightness(image,rate=0):      #https://pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_brightness.html
    brightness_adjusted_img = torchvision.transforms.functional.adjust_brightness(image,rate)
    return brightness_adjusted_img

def compute_adjusted_contrast(image,rate=0):      #https://pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_contrast.html
    contrast_adjusted_img = torchvision.transforms.functional.adjust_contrast(image,rate)
    return contrast_adjusted_img
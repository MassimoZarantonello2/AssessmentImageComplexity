import torchvision

def compute_hue_variation(image,rate = 0):       #https://pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_hue.html
    hue_adjusted_img = torchvision.transforms.functional.adjust_hue(image,rate)
    return hue_adjusted_img

def compute_horizontal_variation(image,rate=0):      #https://pytorch.org/vision/main/generated/torchvision.transforms.RandomHorizontalFlip.html 
    horizontal_adjusted_img = torchvision.transforms.functional.adjust_hue(image,rate)
    return horizontal_adjusted_img

def compute_posterize_variation(image,bits=4):      #https://pytorch.org/vision/main/generated/torchvision.transforms.functional.posterize.html
    posterize_adjusted_img = torchvision.transforms.functional.posterize(image,bits)
    return posterize_adjusted_img

def compute_saturation_variation(image,rate=0):      #https://pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_saturation.html
    saturation_adjusted_img = torchvision.transforms.functional.adjust_saturation(image,rate)
    return saturation_adjusted_img

def compute_solarize_variation(image,threshold=128):      #https://pytorch.org/vision/main/generated/torchvision.transforms.functional.solarize.html
    solarize_adjusted_img = torchvision.transforms.functional.solarize(image,threshold)
    return solarize_adjusted_img
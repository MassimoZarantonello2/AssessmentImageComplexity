import torch
import argparse
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import sys
sys.path.append('./ICNet')
from ICNet import ICNet
import csv
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image

values =[]
class Transform:
	def __init__(self, trans_type):
		self.type = trans_type

	def __call__(self, im):
		if isinstance(im, torch.Tensor):
			raise TypeError(f"pic should be PIL Image. Got {type(im)}")
		global values
		if self.type == 'brightness':
			values = np.linspace(-128, 128, 5)
			im_ycbcr = np.asarray(im.convert('YCbCr'))
			im = []
			for v in values:
				im_ycbcr_ = im_ycbcr.copy().astype(np.float32)
				im_ycbcr_[:,:,0] = np.clip(im_ycbcr_[:,:,0] + v, 0, 255)
				im_ycbcr_ = Image.fromarray(im_ycbcr_.astype(np.uint8), mode='YCbCr').convert('RGB')
				im.append(T.ToTensor()(im_ycbcr_))
		elif self.type == 'contrast':
			values = torch.linspace(0.3, 2, 5)
			im = [TF.adjust_contrast(T.ToTensor()(im), contrast_factor=v) for v in values]
		elif self.type == 'saturation':
			values = torch.linspace(0.3, 2, 5)
			im = [TF.adjust_saturation(T.ToTensor()(im), saturation_factor=v) for v in values]
		elif self.type == 'hue':
			values = [-0.42, -0.19, 0.15, 0.32, 0.49]
			im = im = [TF.adjust_hue(T.ToTensor()(im), hue_factor=v) for v in values]
		elif self.type == 'noise':
			im = T.ToTensor()(im)
			values = torch.linspace(0.01, 0.1, 5)
			im = [im + torch.randn(im.size()[1:])[None] * std for std in values]
		elif self.type == 'hflip':
			im = [TF.hflip(T.ToTensor()(im))]
		elif self.type == 'posterize':
			values = torch.linspace(7, 1, 5, dtype=int)
			im = [T.ToTensor()(TF.posterize(im, bits=v.item())) for v in values]
		elif self.type == 'blur':
			values = torch.linspace(3, 27, 5, dtype=int)
			im = [TF.gaussian_blur(T.ToTensor()(im), kernel_size=v.item()) for v in values] 
		elif self.type == 'jpeg':
			values = torch.linspace(90, 10, 5, dtype=int)
			images = []
			for v in values:
				im.save('tmp.jpg', quality=v.item())
				images.append(T.ToTensor()(Image.open('tmp.jpg')))
			im = images
		return im

def blend(ori_img, ic_img, alpha = 0.8, cm = plt.get_cmap("magma")):
    cm_ic_map = cm(ic_img) 
    heatmap = Image.fromarray((cm_ic_map[:, :, -2::-1]*255).astype(np.uint8))
    ori_img = Image.fromarray(ori_img)
    blend = Image.blend(ori_img,heatmap,alpha=alpha)
    blend = np.array(blend)
    return blend

        
def infer_one_image(image):   #mette il modello in modalità di valutazione 
    with torch.no_grad():      #disabilita il calcolo dei gradienti
        #ori_height = image.height
        #ori_width = image.width
        img = image.to(device)   
        img = img.unsqueeze(0)    #https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch
        ic_score, ic_map = model(img)   #esegue la predizione ritornando lo score della complessità e la mappa
        ic_score = ic_score.item() 
        #ic_map = F.interpolate(ic_map, (ori_height, ori_width), mode = 'bilinear')
        
        ## gene blend map
        #ic_map_img = (ic_map * 255).round().squeeze().detach().cpu().numpy().astype('uint8')
        #blend_img = blend(np.array(image), ic_map_img)
        return ic_score

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='test transformations')
	parser.add_argument("type", choices=['brightness', 'contrast', 'saturation',
										'hue', 'noise', 'hflip', 'posterize',
										'blur', 'jpeg'], default='noise')
	parser.add_argument("-iph",dest="path",default='./my_images/')
	args = parser.parse_args()
 
	inference_transform = T.Compose([T.Resize((512,512)),
								Transform(args.type),
								T.Lambda(lambda x: torch.stack(x)),
								T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
								])
	model = ICNet()
	model.load_state_dict(torch.load('./checkpoint/ck.pth',map_location=torch.device('cpu')))
	model.eval()
	device = torch.device(0)
	model.to(device) 
	operation = args.type
	with open(f'transformation/files_csv/{operation}.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		image_list = os.listdir(args.path)
		for image_name in image_list:
			complexities = []
			image = Image.open(args.path + image_name)
			transformed_images = inference_transform(image)
			for image in transformed_images:
				complexities.append(str(infer_one_image(image)))
			writerow = [image_name] + complexities
			writer.writerow(writerow)
			complexities = []
import torch
import argparse
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import semanticPreservingTransformation as SPT
from PIL import Image
from ICNet import gene

from PIL import Image

class Transform:
	def __init__(self, trans_type):
		self.type = trans_type

	def __call__(self, im):
		if isinstance(im, torch.Tensor):
			raise TypeError(f"pic should be PIL Image. Got {type(im)}")

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



parser = argparse.ArgumentParser(description='test transformations')
parser.add_argument("type", choices=['brightness', 'contrast', 'saturation',
									 'hue', 'noise', 'hflip', 'posterize',
									 'blur', 'jpeg'], default='noise')
args = parser.parse_args()

image_name = 'scene_Twin_Tower,_Malaysia.jpg'

im = Image.open(f'.\my_images\{image_name}')

inference_transform = T.Compose([T.Resize((512,512)),
								 Transform(args.type),
								 T.Lambda(lambda x: torch.stack(x)),
								 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
								])

im = inference_transform(im)
for image in im:
	image = Image.fromarray((image.permute(1,2,0).numpy() * 255).astype(np.uint8))
	gene.infer_one_image(image, 'image_name.jpg')
		
from ICNet import ICNet
import argparse
import os 
import torch
import cv2
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type = str, default = './example')
parser.add_argument('-o', '--output', type = str, default = './out')
parser.add_argument('-d', '--device', type = int, default=0)

def blend(ori_img, ic_img, alpha = 0.8, cm = plt.get_cmap("magma")):
    cm_ic_map = cm(ic_img) 
    heatmap = Image.fromarray((cm_ic_map[:, :, -2::-1]*255).astype(np.uint8))
    ori_img = Image.fromarray(ori_img)
    blend = Image.blend(ori_img,heatmap,alpha=alpha)
    blend = np.array(blend)
    return blend

        
def infer_one_image(img_path):
    with torch.no_grad():      #disabilita il calcolo dei gradienti
        ori_img = Image.open(img_path).convert("RGB")
        ori_height = ori_img.height
        ori_width = ori_img.width
        img = inference_transform(ori_img)
        img = img.to(device)    
        img = img.unsqueeze(0)    #https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch
        ic_score, ic_map = model(img)   #esegue la predizione ritornando lo score della complessità e la mappa
        ic_score = ic_score.item() 
        ic_map = F.interpolate(ic_map, (ori_height, ori_width), mode = 'bilinear')
        
        ## gene ic map
        ic_map_np = ic_map.squeeze().detach().cpu().numpy()     #carica la mappa su CPU e la trasforma in numpy
        out_ic_map_name = os.path.basename(img_path).split('.')[0] + '_' + str(ic_score)[:7] + '.npy'   #salva la mappa con il nome dell'immagine e lo score
        out_ic_map_path = os.path.join(args.output, out_ic_map_name)    
        np.save(out_ic_map_path, ic_map_np)
        
        ## gene blend map
        ic_map_img = (ic_map * 255).round().squeeze().detach().cpu().numpy().astype('uint8')
        blend_img = blend(np.array(ori_img), ic_map_img)
        out_blend_img_name = os.path.basename(img_path).split('.')[0]  + '.png'
        out_blend_img_path = os.path.join(args.output, out_blend_img_name)
        cv2.imwrite(out_blend_img_path, blend_img)
        return ic_score

    
    
def infer_directory(img_dir):
    imgs = os.listdir(img_dir)
    for img in tqdm(imgs):      #esegue infer_one_image su tutte le immagini della cartella
        img_path = os.path.join(img_dir, img)
        infer_one_image(img_path)

if __name__ == "__main__":
    args  = parser.parse_args()
    
    model = ICNet()     #inizzializza il medello ICNet
    model.load_state_dict(torch.load('./checkpoint/ck.pth',map_location=torch.device('cpu')))      #carica i pesi del modello
    model.eval()    #mette il modello in modalità di valutazione
    device = torch.device(args.device)   #seleziona la GPU
    model.to(device)    #sposta il modello sulla GPU
    
    inference_transform = transforms.Compose([      #esegue delle trasformazioni sull'immagine
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if os.path.isfile(args.input):      #se l'input è un file
        infer_one_image(args.input)
    else:
        infer_directory(args.input)


    







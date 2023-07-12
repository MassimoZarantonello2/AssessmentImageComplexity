import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from scipy.stats import pearsonr, spearmanr

from Net import ICNet
from dataset import ic_dataset


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type = str, default = './example')
parser.add_argument('-o', '--output', type = str, default = './out')
parser.add_argument('-d', '--device', type = int, default=0)
parser.add_argument("--type", dest = 'type', choices=['brightness', 'contrast', 'saturation',
                                    'hue', 'noise', 'hflip', 'posterize',
                                    'blur', 'jpeg'], default='none')


def evaInfo(score, label):
    score = np.array(score)
    label = np.array(label)

    RMAE = np.sqrt(np.abs(score - label).mean())    #calcola la radice della media degli errori assoluti
    RMSE = np.sqrt(np.mean(np.abs(score - label) ** 2))   #calcola la radice della media degli errori quadrati
    Pearson = pearsonr(label, score)[0]    #calcola il coefficiente di correlazione di Pearson
    Spearmanr = spearmanr(label, score)[0]    #calcola il coefficiente di correlazione di Spearman

    info = 'RMSE : {:.4f}, RMAE : {:.4f}, Pearsonr : {:.4f}, Spearmanr : {:.4f}'.format(
               RMSE,  RMAE, Pearson, Spearmanr) 

    return info


class ICNetModel():
    def __init__(self, device, args):
        self.args = args
        self.model = ICNet()
        self.model.load_state_dict(torch.load('./checkpoint/ck.pth',map_location=torch.device('cpu')))
        self.model.eval()
        self.model.to(device)

        self.preprocess = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]
                                 )
            ])

    @torch.no_grad()
    def infer_one_image(self, img_path):
        ori_img = Image.open(img_path).convert("RGB")
        ori_height = ori_img.height
        ori_width = ori_img.width
        img = self.preprocess(ori_img)
        img = img.to(device)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        ic_score, _ = self.model(img)
        return ic_score

    def infer_dataset(self, data_file, image_path):
        dataset = ic_dataset(data_file, image_path)
        test_labels = torch.zeros(len(dataset))
        test_pred = torch.zeros(len(dataset))

        for i, r in enumerate(tqdm(dataset.txt_lines)):
            filename, tmp = r.split("  ")
            test_labels[i] = float(tmp)
            test_pred[i] = self.infer_one_image(os.path.join(image_path, filename))

        info = evaInfo(test_pred, test_labels)
        torch.save({"gt": test_labels, "pred": test_pred}, "outputs.pth")
        with open("results.txt", "w") as f:
             f.write(info)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args  = parser.parse_args()

    model = ICNetModel(device, args)
    model.infer_dataset(*args.input.split(","))
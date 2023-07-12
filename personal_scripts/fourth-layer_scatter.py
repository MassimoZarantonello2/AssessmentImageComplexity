import os
from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt

def save_graph(x,y,title):
    
    area = np.array(range(len(x)))
    plt.scatter(area, x, label = 'Ground-Truth', color = 'purple',s=2)
    plt.scatter(area, y, label = 'Values', color = 'lightblue',s=2)
    plt.title(title)
    plt.legend()
    plt.savefig('./' + title + '.png')
    plt.clf()
    
    

gt_path = './IC9600/parsed_files/test_parsed.txt'
gt_scores = []
res_scores = []

with open(gt_path, 'r') as gt:
    lines = gt.readlines()
    for line in lines:
        _,gt_score = line.split(' ')
        gt_scores.append(float(gt_score))
'''
res_path = './ResNet18/test_results/new_IC9600_ResNet18_fourth_layer.txt'

with open(res_path, 'r') as res:
    lines = res.readlines()
    for line in lines:
        _,res_score = line.split(' ')
        res_scores.append(float(res_score))
save_graph(gt_scores,res_scores,'ResNet18_fourth_layer')
'''      
vgg_path = './VGG-16/test_results/Savoias_VGG16_fourth_layer_sorted.txt'

with open(vgg_path, 'r') as res:
    lines = res.readlines()
    for line in lines:
        _,res_score = line.split(' ')
        res_scores.append(float(res_score))
save_graph(gt_scores,res_scores,'VGG16_fourth_layer') 

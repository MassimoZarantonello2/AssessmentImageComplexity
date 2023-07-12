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
    plt.savefig('./graphs/difference_graphs/' + title + '.png')
    plt.clf()
    
    

gt_path = './IC9600/parsed_files/test_parsed.txt'
gt_scores = []

with open(gt_path, 'r') as gt:
    lines = gt.readlines()
    for line in lines:
        _,gt_score = line.split(' ')
        gt_scores.append(float(gt_score))

tras_categories = [[],[],[],[],[]]
tras_path = './transformation/aj_files_csv/'

tras_files = os.listdir(tras_path)

for file in tras_files:
    name = file.split('.')[0].split('_')[1]
    file_path = tras_path + file
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
                if name == 'hflip':
                    tras_categories[0].append(float(row[1]))
                else:
                    tras_categories[0].append(float(row[1]))
                    tras_categories[1].append(float(row[2]))
                    tras_categories[2].append(float(row[3]))
                    tras_categories[3].append(float(row[4]))
                    tras_categories[4].append(float(row[5]))
                
        intensity = int(input(f'Enter intensity of {name}: '))
        save_graph(gt_scores,tras_categories[intensity],name + '_' + str(intensity))
        tras_categories = [[],[],[],[],[]]
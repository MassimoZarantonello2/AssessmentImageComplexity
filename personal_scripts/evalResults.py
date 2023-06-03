import numpy as np
from scipy.stats import pearsonr, spearmanr
import os
import pandas as pd
import time
import csv

def evaInfo(score,label):
    
    score = np.array(score)
    score = score.astype(float)
    label = np.array(label)
    label = label.astype(float)

    RMAE = np.sqrt(np.abs(score-label).mean())    #calcola la radice della media degli errori assoluti
    RMSE = np.sqrt(np.mean(np.abs(score-label) ** 2))   #calcola la radice della media degli errori quadrati
    Pearson = pearsonr(label, score)[0]    #calcola il coefficiente di correlazione di Pearson
    Spearmanr = spearmanr(label, score)[0]    #calcola il coefficiente di correlazione di Spearman

    return RMAE,RMSE,Pearson,Spearmanr

def get_all_data(path):         #funzione che ritorna una lista di label e una lista di score se sono nel formato "label score" !!separati da uno spazio
    labels = []
    scores = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name,cpl_score = line.split(" ", 1)
            labels.append(name)
            scores.append(cpl_score[:len(cpl_score)-1])
    return labels,scores

def get_partial_data(path,GT_labels):            #funzione che ritorna una lista di 
    ICNet_test_results = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name,cpl_score = line.split(" ", 1)
            if name in GT_labels:
                ICNet_test_results.append(cpl_score[:len(cpl_score)-1])
    return ICNet_test_results

def get_category_data(path):
    last_category = ''
    categories_score = []
    category_score = []
    i = 0
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            category,name_score = line.split('_', 1)
            score = name_score.split(" ", 1)[1][:len(name_score.split(" ", 1)[1])-1]
            
            if(category == last_category):
                category_score.append(score)
            else:
                categories_score.append(category_score)
                category_score = [score]
                last_category = category
                i += 1
    categories_score.append(category_score)
    return categories_score[1:]

def get_Savoias_GT_Scores():
    GT_scores = []
    GT_path = "./Savoias-Dataset-master/GroundTruth/xlsx/"
    GT_files = os.listdir(GT_path)
    for GT_file in GT_files:
        df = pd.read_excel(GT_path+GT_file)
        #read the first column of the excel file
        for index, row in df.iterrows():
            GT_scores.append(row[0]/100)
    return GT_scores

def csv_output(info,info2):
    metrics = ['RMSE','RMAE','Pearsonr','Spearman']
    with open('./evalResults_info.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metrics", info2])
        for i in range(len(metrics)):
            writer.writerow([metrics[i],info[i]])

if __name__ == "__main__":
    '''ResNet-18 with IC9600'''
    resnet_res = './ResNet18/test_results/new_IC9600_ResNet18_fourth_layer.txt'
    IC_GT = './IC9600/parsed_files/test_and_train_parsed.txt'
    _,resnet_scores = get_all_data(resnet_res)
    _,IC_GT_labels = get_all_data(IC_GT)
    csv_output(evaInfo(resnet_scores,IC_GT_labels),'ResNet-18 with IC9600')
    
    '''ResNet-18 with Savoias'''
    resnet_res = './ResNet18/test_results/Savoias_ResNet18_fourth_layer.txt'
    Savoias_GT = './Savoias-Dataset-master/GroundTruth/xlsx/'
    _,resnet_scores = get_all_data(resnet_res)
    Savoias_GT_scores = get_Savoias_GT_Scores()
    csv_output(evaInfo(resnet_scores,Savoias_GT_scores),'ResNet-18 with Savoias')
    
    '''VGG-16 with IC9600'''
    vgg_res = './VGG-16/test_results/IC9600_VGG-16_fourth_layer.txt'
    IC_GT = './IC9600/parsed_files/test_parsed.txt'
    label,vgg_scores = get_all_data(vgg_res)
    _,IC_GT_labels = get_all_data(IC_GT)
    csv_output(evaInfo(vgg_scores,IC_GT_labels),'VGG-16 with IC9600')
    
    '''VGG-16 with Savoias'''
    vgg_res = './VGG-16/test_results/Savoias_VGG16_fourth_layer_sorted.txt'
    _,vgg_scores = get_all_data(vgg_res)
    csv_output(evaInfo(vgg_scores,Savoias_GT_scores),'VGG-16 with Savoias')
    
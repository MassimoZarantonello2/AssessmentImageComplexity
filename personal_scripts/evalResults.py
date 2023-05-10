import numpy as np
from scipy.stats import pearsonr, spearmanr
import os
import pandas as pd
import time

def evaInfo(score,label):
    
    score = np.array(score)
    score = score.astype(float)
    label = np.array(label)
    label = label.astype(float)

    RMAE = np.sqrt(np.abs(score-label).mean())    #calcola la radice della media degli errori assoluti
    RMSE = np.sqrt(np.mean(np.abs(score-label) ** 2))   #calcola la radice della media degli errori quadrati
    Pearson = pearsonr(label, score)[0]    #calcola il coefficiente di correlazione di Pearson
    Spearmanr = spearmanr(label, score)[0]    #calcola il coefficiente di correlazione di Spearman

    info = ' RMSE : {:.4f} ,   RMAE : {:.4f} ,   Pearsonr : {:.4f} ,   Spearmanr : {:.4f}'.format(
               RMSE,  RMAE, Pearson, Spearmanr) 

    return info

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
        

if __name__ == "__main__":
    #GT_category_scores = get_category_data("./IC9600/parsed_files/test_and_train_parsed.txt")
    #ICNet_category_scores = get_category_data("./IC9600/ICNet_results.txt")

    '''IC9600 with VGG16'''
    _,IC9600_GT = get_all_data('./IC9600/parsed_files/test_and_train_parsed.txt')
    _,ICNet_IC9600_scores = get_all_data('./VGG-16/test_results/VGG_ResNet18_fourth_layer_normalized.txt')
    print('\nIC9600 with VGG16')
    print(evaInfo(IC9600_GT,ICNet_IC9600_scores))
    print('---------------------------------\n')
    time.sleep(1)
    
    '''IC9600 with ResNet'''
    _,IC9600_GT = get_all_data('./IC9600/parsed_files/test_and_train_parsed.txt')
    _,ICNet_IC9600_scores = get_all_data('./ResNet18/test_results/new_IC9600_ResNet18_fourth_layer_normalized.txt')
    print('\nIC9600 with VGG16')
    print(evaInfo(IC9600_GT,ICNet_IC9600_scores))
    print('---------------------------------\n')
    time.sleep(1)
    
    '''SAVOIAS with ResNet'''
    _,IC9600_GT = get_all_data('./IC9600/parsed_files/test_and_train_parsed.txt')
    _,ICNet_IC9600_scores = get_all_data('./ResNet18/test_results/new_IC9600_ResNet18_fourth_layer_normalized.txt')
    print('\nIC9600 with VGG16')
    print(evaInfo(IC9600_GT,ICNet_IC9600_scores))
    print('---------------------------------\n')
    time.sleep(1)
    
    '''SAVOIAS with VGG16'''
    _,IC9600_GT = get_all_data('./IC9600/parsed_files/test_and_train_parsed.txt')
    _,ICNet_IC9600_scores = get_all_data('./ResNet18/test_results/new_IC9600_ResNet18_fourth_layer_normalized.txt')
    print('\nIC9600 with VGG16')
    print(evaInfo(IC9600_GT,ICNet_IC9600_scores))
    print('---------------------------------\n')
    time.sleep(1)
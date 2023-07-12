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

def get_all_data2(path):        #funzione che ritorna una lista di label e una lista di score se sono nel formato "label_score" !!separati da un underscore
    labels = []
    scores = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name,cpl_score = line.split(" ", 1)
            labels.append(name)
            scores.append(float(cpl_score[:len(cpl_score)-1]))
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
    _,ICNet_savoias = get_all_data2("./new_my_ICNet_SAVOIAS_results.txt")
    scores = get_Savoias_GT_Scores()
    print(evaInfo(ICNet_savoias,scores))
import csv
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr

main_path = './transformation/aj_files_csv/'
test_ICNet_results_path = './ICNet/test_results/test_ICNet_results.txt'
write_csv_results = './transformation/evaluated_results.csv'
csv_files = os.listdir(main_path)
test_dict = {}
scores = [[],[],[],[],[]]
GT_score = []
metrics = ['MAE','RMSE','Spearmans','Pearsons']

def evaInfo(score,label):
    
    score = np.array(score)
    score = score.astype(float)
    label = np.array(label)
    label = label.astype(float)

    RMAE = np.sqrt(np.abs(score-label).mean())    #calcola la radice della media degli errori assoluti
    RMSE = np.sqrt(np.mean(np.abs(score-label) ** 2))   #calcola la radice della media degli errori quadrati
    Pearson = pearsonr(label, score)[0]    #calcola il coefficiente di correlazione di Pearson
    Spearmanr = spearmanr(label, score)[0]    #calcola il coefficiente di correlazione di Spearman 

    return RMAE,RMSE,Spearmanr,Pearson

with open(test_ICNet_results_path, 'r') as f:
    for line in f:
        name,score = line.split(' ')
        GT_score.append(float(score))
        
for csv_file in csv_files:
    with open(main_path + csv_file,'r') as read_csv_file:
        with open(write_csv_results,'a') as write_csv_file:
            csv_writer = csv.writer(write_csv_file)
            csv_reader = csv.reader(read_csv_file, delimiter=',')
            values = next(csv_reader)
            csv_writer.writerow([values[0]]+metrics)
            eval_scores = [[],[],[],[],[]]
            for line in csv_reader:
                scores = line[1:]
                for i in range(len(scores)):
                    eval_scores[i].append(float(scores[i]))
            for i in range(len(eval_scores)):
                if(eval_scores[i] != []):
                    evaluations = evaInfo(GT_score,eval_scores[i])
                    write_value = [values[i+1]]
                    for j in range(len(evaluations)):
                        write_value.append(evaluations[j])
                    csv_writer.writerow(write_value)        
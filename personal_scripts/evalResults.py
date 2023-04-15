import numpy as np
from scipy.stats import pearsonr, spearmanr

def difference(score,label):
    difference_res = []
    for i in range(len(score)):
        x = float(score[i])
        y = float(label[i])
        difference_res.append(x-y)
    return difference_res
        

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

def get_scores(path):
    lista = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name,cpl_score = line.split(" ", 1)
            lista.append(cpl_score[:len(cpl_score)-1])
    return lista

def get_labels(path):
    lista = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name,cpl_score = line.split(" ", 1)
            lista.append(name)
    return lista

if __name__ == "__main__":
    GT_results = get_scores("./IC9600/GT_IC9600.txt")
    ICNet_results = get_scores("./IC9600/IC9600_results.txt")
    
    print(evaInfo(ICNet_results,GT_results))
    
import numpy as np
from scipy.stats import pearsonr, spearmanr

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

def get_all_data(path):
    labels = []
    scores = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name,cpl_score = line.split(" ", 1)
            labels.append(name)
            scores.append(cpl_score[:len(cpl_score)-1])
    return labels,scores

def get_partial_data(path,GT_labels):
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

if __name__ == "__main__":
    #GT_category_scores = get_category_data("./IC9600/parsed_files/test_and_train_parsed.txt")
    #ICNet_category_scores = get_category_data("./IC9600/ICNet_results.txt")
    labels,GT_scores = get_all_data("./IC9600/parsed_files/test_and_train_parsed.txt")
    labels,ResNet18_scores = get_all_data("./ResNet18/ResNet18_GT.txt")
    print(evaInfo(ResNet18_scores,GT_scores))
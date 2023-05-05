import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from scipy.stats import pearsonr, spearmanr

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
    

def evaInfo(score,label):
    score = np.array(score)
    label = np.array(label)

    RMAE = np.sqrt(np.abs(score - label).mean())    #calcola la radice della media degli errori assoluti
    RMSE = np.sqrt(np.mean(np.abs(score - label) ** 2))   #calcola la radice della media degli errori quadrati
    Pearson = pearsonr(label, score)[0]    #calcola il coefficiente di correlazione di Pearson
    Spearmanr = spearmanr(label, score)[0]    #calcola il coefficiente di correlazione di Spearman

    info = ' RMSE : {:.4f} ,   RMAE : {:.4f} ,   Pearsonr : {:.4f} ,   Spearmanr : {:.4f}'.format(
               RMSE,  RMAE, Pearson, Spearmanr) 

    return info


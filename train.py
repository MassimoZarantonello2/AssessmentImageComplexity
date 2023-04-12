
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import WarmUpLR, evaInfo
from options import args
import os
from torch import optim
from dataset import ic_dataset
from ICNet import ICNet


def train(epoch):
    model.train()
    for batch_index, (image,label,_) in enumerate(trainDataLoader):        
        image = image.to(device)        #sporta l'immagine e l'etichetta sulla GPU
        label = label.to(device)       
        Opmimizer.zero_grad()        #azzera i gradienti
        score1, cly_map = model(image)   #passa l'immagine alla rete neurale e ottiene la mappa di complessità e lo score
        score2 = cly_map.mean(axis = (1,2,3)) #calcola lo score come la media della mappa di complessità su tutti i pixel
        loss1 = loss_function(score1,label) #calcola la loss
        loss2 = loss_function(score2,label) #calcola la loss della media della mappa di complessità
        loss = 0.9*loss1 + 0.1*loss2    #calcola la loss finale come combinazione lineare delle due loss
        loss.backward()     #calcola i gradienti ??
        Opmimizer.step()        #aggiorna i pesi della rete neurale
        if epoch <= args.warm:      #se l'epoch fa parte della fase di warmup allora aggiorna il learning rate
            Warmup_scheduler.step()

        if (batch_index+1) % (len(trainDataLoader) // 3) == 0:      #stampa la loss ogni 1/3 dell'epoch
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tloss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                Opmimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.batch_size + len(image),
                total_samples=len(trainDataLoader.dataset)
            ))


def evaluation():
    model.eval()    #mette la rete neurale in modalità di valutazione
    all_scores = []
    all_labels = []
    for (image, label, _) in testDataLoader:
        image = image.to(device)    #sporta l'immagine e l'etichetta sulla GPU
        label = label.to(device)
        with torch.no_grad():    #disabilita il calcolo dei gradienti per velocizzare il processo di valutazione
            score, _= model(image)
            all_scores += score.tolist()
            all_labels += label.tolist()
    info = evaInfo(score=all_scores, label=all_labels)  #calcola le metriche di valutazione
    print(info + '\n')




if __name__ == "__main__":
    
    trainTransform = transforms.Compose([               #applica delle trasformate all'immagine prima di passarla alla rete neurale
    transforms.Resize((args.image_size, args.image_size)),      #ridimensiona l'immagine
    transforms.RandomHorizontalFlip(),                     #ruota l'immagine sull'asse orrizontale
    transforms.ToTensor(),                                 #trasforma l'immagine in un tensore
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #normalizza l'immagine
])

    testTransform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
  
    trainDataset = ic_dataset(          #crea un dataset che carica le immagini e le etichette
        txt_path ="./IC9600/train.txt",
        img_path = "./IC9600/images/",
        transform = trainTransform
    )

    
    trainDataLoader = DataLoader(trainDataset,      #crea un dataloader che carica i dati del dataset 
                             batch_size=args.batch_size,        #numero di immagini che carica per volta prima di aggiornare i pesi
                             num_workers=args.num_workers,
                             shuffle=True
                             )

    testDataset = ic_dataset(
        txt_path= "./IC9600/test.txt",
        img_path = "./IC9600/images/",
        transform=testTransform
    )
    
    testDataLoader = DataLoader(testDataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False
                            )
    if not os.path.exists(args.ck_save_dir):        #controlla se esiste una cartella dove salvare i checkpoint altrimenti la crea
        os.mkdir(args.ck_save_dir)
    
    model = ICNet()         #inizzializza il modello della rete neurale ICNet
    
    device = torch.device("cuda:{}".format(args.gpu_id))    #seleziona la gpu con cui lavorare
    model.to(device)

    loss_function = nn.MSELoss()    #definisce la funzione di loss
    
    # optimize
    params = model.parameters()
    Opmimizer = optim.SGD(params, lr =args.lr,momentum=0.9,weight_decay=args.weight_decay)  #definisce l'ottimizzatore utilizzando la funzione SGD (Stochastic Gradient Descent)
    Scheduler = optim.lr_scheduler.MultiStepLR(Opmimizer,milestones=args.milestone,gamma = args.lr_decay_rate)  #definisce la funzione che riduce il learning rate
    iter_per_epoch = len(trainDataLoader)       #numero di volte che un immagine ha la possibilità di essere vista dalla rete neurale
    if args.warm > 0:   #se il learning rate è basso viene usato un warmup(riscaldamento) per poi passare al learning rate definitivo
        Warmup_scheduler = WarmUpLR(Opmimizer,iter_per_epoch*args.warm)
    
    # running
    for epoch in range(1, args.epoch+1):
        train(epoch)
        if epoch > args.warm:
            Scheduler.step(epoch)
        evaluation()
        torch.save(model.state_dict(), os.path.join(args.ck_save_dir,'ck_{}.pth'.format(epoch)))

    









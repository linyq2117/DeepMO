import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import math
import sklearn.preprocessing as sk
import seaborn as sns
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
#import utils
#from myutils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
#from mymetrics import AverageNonzeroTripletsMetric
#from utils import RandomNegativeTripletSelector

from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import random
from random import randint
from sklearn.model_selection import StratifiedKFold,RepeatedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp

import numpy as np
import warnings

import math
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
warnings.filterwarnings("ignore")
#save_results_to = '/home/winston/MOLI/res/cnv_6index/nofs/'
save_results_to = 'F:/MOLI/result/all/'
#torch.manual_seed(42)
random.seed(42)

max_iter = 1


TCGAEx = pd.read_csv("F:/MOLI/data/brca_cnv_subtype.csv", 
                    index_col=0, header=None)
TCGAEx=TCGAEx.values
TCGAEx=TCGAEx[:,0:606]
#print(TCGAEx)

label=TCGAEx[0,:]
data=TCGAEx[1:,:]
#print(label)

label[np.where(label=='lminalA')]=0
label[np.where(label=='lminalB')]=1
label[np.where(label=='TNBC')]=2
label[np.where(label=='ERBB2')]=3
label[np.where(label=='normal')]=4

data=data.T

Y=np.array(label.astype(int))
min_max_scaler = preprocessing.MinMaxScaler()
x1 = min_max_scaler.fit_transform(data)
TCGAE = SelectKBest(chi2, k=5000).fit_transform(x1, Y)
'''
ls_mb_size = [13, 36, 64]
ls_h_dim = [1024, 256, 128, 512, 64, 16]
#ls_h_dim = [32, 16, 8, 4]
ls_marg = [0.5, 1, 1.5, 2, 2.5, 3]
ls_lr = [0.5, 0.1, 0.05, 0.01, 0.001, 0.005, 0.0005, 0.0001,0.00005, 0.00001]
ls_epoch = [20, 50, 90, 100]
ls_rate = [0.3, 0.4, 0.5]
ls_wd = [0.1, 0.001, 0.0001]
ls_lam = [0.1, 0.5, 0.01, 0.05, 0.001, 0.005]
'''
ls_mb_size = [64]
ls_h_dim = [512]

ls_lr = [0.01]
ls_epoch = [100]
ls_rate = [0.5]
ls_wd = [0.001]


skf = StratifiedKFold(n_splits=5, random_state=None,shuffle=True)

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
for iters in range(max_iter):
    #print('iters:',iters)
    k = 0
    mbs = random.choice(ls_mb_size)
    hdm = random.choice(ls_h_dim)
    #mrg = random.choice(ls_marg)
    lre = random.choice(ls_lr)
    lrCL = random.choice(ls_lr)
    epch = random.choice(ls_epoch)
    rate = random.choice(ls_rate)
    wd = random.choice(ls_wd)   
    #lam = random.choice(ls_lam)

    costtr_all=[]
    costts_all=[]
    auctr_all=[]
    aucts_all=[]
    acctr_all=[]
    accts_all=[]
    sentr_all=[]
    sents_all=[]
    spetr_all=[]
    spets_all=[]
    # 画平均ROC曲线的两个参数

    mean_tpr = 0.0              # 用来记录画平均ROC曲线的信息

    mean_fpr = np.linspace(0, 1, 100)

    #cnt = 0

    for repeat in range(10):
        for train_index, test_index in skf.split(TCGAE, Y.astype('int')):
            
            #if(k%5==0):
            k = k + 1
            X_trainE = TCGAE[train_index,:]
            X_testE =  TCGAE[test_index,:]
            y_trainE = Y[train_index]
            y_testE = Y[test_index]
            
            
            TX_testE = torch.FloatTensor(X_testE)
            ty_testE = torch.FloatTensor(y_testE.astype(int))
            
            #Train
            class_sample_count = np.array([len(np.where(y_trainE==t)[0]) for t in np.unique(y_trainE)])
            #print(class_sample_count)
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in y_trainE])

            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

            mb_size = mbs

            trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_trainE), torch.FloatTensor(y_trainE.astype(int)))

            trainLoader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=mb_size, shuffle=False, num_workers=0, sampler = sampler)

            n_sampE, IE_dim = X_trainE.shape

            h_dim = hdm
            Z_in = h_dim
           
            lrE = lre
            epoch = epch

            costtr = []
            auctr = []
            costts = []
            aucts = []
            acctr=[]
            accts=[]
            acc0tr=[]
            acc0ts=[]
            sentr=[]
            sents=[]
            spetr=[]
            spets=[]

     
            
            class AEE(nn.Module):
                def __init__(self):
                    super(AEE, self).__init__()
                    self.EnE = torch.nn.Sequential(
                        nn.Linear(IE_dim, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.ReLU(),
                        nn.Dropout())
                def forward(self, x):
                    output = self.EnE(x)
                    return output  

            class OnlineTriplet(nn.Module):
                def __init__(self, marg, triplet_selector):
                    super(OnlineTriplet, self).__init__()
                    self.marg = marg
                    self.triplet_selector = triplet_selector
                def forward(self, embeddings, target):
                    triplets = self.triplet_selector.get_triplets(embeddings, target)
                    return triplets

            class OnlineTestTriplet(nn.Module):
                def __init__(self, marg, triplet_selector):
                    super(OnlineTestTriplet, self).__init__()
                    self.marg = marg
                    self.triplet_selector = triplet_selector
                def forward(self, embeddings, target):
                    triplets = self.triplet_selector.get_triplets(embeddings, target)
                    return triplets    

            class Classifier(nn.Module):
                def __init__(self):
                    super(Classifier, self).__init__()
                    self.FC = torch.nn.Sequential(
                        nn.Linear(Z_in, 5),
                        nn.Dropout(rate))
                def forward(self, x):
                    return F.softmax(self.FC(x))
                    #return self.FC(x)

            torch.cuda.manual_seed_all(42)

            AutoencoderE = AEE()


            solverE = optim.Adagrad(AutoencoderE.parameters(), lr=lrE)

 

            Clas = Classifier()
            SolverClass = optim.Adagrad(Clas.parameters(), lr=lrCL, weight_decay = wd)
            #C_loss = torch.nn.BCELoss()
            C_loss = torch.nn.CrossEntropyLoss()
            print('epoch_all',epoch)
            for it in range(epoch):
                #print('epoch:',it)
                epoch_cost4 = 0
                epoch_cost3 = []
                p_real=[]
                p_pred=[]
                n_real=[]
                n_pred=[]
                p_realt=[]
                p_predt=[]
                n_realt=[]
                n_predt=[]
            
                correct = torch.zeros(1).squeeze()
                total = torch.zeros(1).squeeze()
                correctt = torch.zeros(1).squeeze()
                totalt = torch.zeros(1).squeeze()
                num_minibatches = int(n_sampE / mb_size) 

                for i, (dataE, target) in enumerate(trainLoader):
                    
                    flag = 0
                    AutoencoderE.train()

                    Clas.train()

                    #if torch.mean(target)!=0. and torch.mean(target)!=1.: 
                    ZEX = AutoencoderE(dataE)
                    Pred = Clas(ZEX)
                    #print(Pred)

                  
                    loss=C_loss(Pred,target.view(-1,1).long().squeeze())
                    y_true = target.view(-1,1)
                    y_pred = Pred
                    
                    prediction = torch.argmax(y_pred, 1)
                    correct += (prediction == target.long().squeeze()).sum().float()
                    total += len(target)
                                       
                    solverE.zero_grad()
                    SolverClass.zero_grad()

                    loss.backward()

                    solverE.step()
                    SolverClass.step()

                    epoch_cost4 = epoch_cost4 + (loss / num_minibatches)
                  
                flag = 1

                if flag == 1:
                    costtr.append(torch.mean(epoch_cost4))
                    
                    acc=(correct/total).detach().numpy()
                    acctr.append(acc)
                if(it==epoch-1):
                    print('acctrain:',acc)
                   
                    
                with torch.no_grad():

                    AutoencoderE.eval()
                    Clas.eval()

                    ZET = AutoencoderE(TX_testE)
                    PredT = Clas(ZET)

                   
                    lossT=C_loss(PredT,ty_testE.view(-1,1).long().squeeze())
                    y_truet = ty_testE.view(-1,1)
                    y_predt = PredT

                    predictiont = torch.argmax(y_predt, 1)
                    correctt += (predictiont == y_truet.long().squeeze()).sum().float()
                    totalt += len(y_truet)
                   

                    
                   
                    
                
                if(it==epoch-1):
                   
                    print('acctest:',acc)
                   

            acctr_all.append(acctr)
            accts_all.append(accts)
            costtr_all.append(costtr)
            costts_all.append(costts)

            if k%50==0:
               
                costtr_all=np.array(costtr_all)
                costts_all=np.array(costts_all)
                costtr5=sum(costtr_all)/k
                costts5=sum(costts_all)/k
                print('cost200:',costts5[-1])
                print('costmin:',min(costts5))
                acctr_all=np.array(acctr_all)
                accts_all=np.array(accts_all)
                acctr5=sum(acctr_all)/k
                accts5=sum(accts_all)/k
                print('acc200:',accts5[-1])
                print('accmax:',max(accts5))
                print('cost:',costts5)
                print('acc:',accts5)
                
                plt.plot(np.squeeze(costtr5), '-r',np.squeeze(costts5), '-b')
                plt.ylabel('Total cost')
                plt.xlabel('epoch')

                title = 'cnv_ Cost(5class)-skf10   mb_size = {},  h_dim = {} , lrE = {}, epoch = {}, rate = {}, wd = {}, lrCL = {}'.\
                              format( mbs, hdm,  lre, epch, rate, wd, lrCL)

                #title = 'Cost  iter = {}, fold = {}, mb_size = {},  h_dim = {}, marg = {}, lrE = {}, epoch = {}, rate = {}, wd = {}, lrCL = {}, lam = {}'.\
                #              format(iters, k, mbs, hdm, mrg, lre, epch, rate, wd, lrCL, lam)

                plt.suptitle(title)
                plt.savefig(save_results_to + title + '.png', dpi = 150)
                plt.close()
               
                plt.plot(np.squeeze(acctr5), '-r',np.squeeze(accts5), '-b')
                plt.ylabel('Accuracy')
                plt.xlabel('epoch')

                title = 'cnv_ Accuracy(A=1,B=0)-skf10   mb_size = {},  h_dim = {},  lrE = {}, epoch = {}, rate = {}, wd = {}, lrCL = {}, '.\
                              format( mbs, hdm, lre, epch, rate, wd, lrCL)        

                plt.suptitle(title)
                plt.savefig(save_results_to + title + '.png', dpi = 150)
                plt.close()

               

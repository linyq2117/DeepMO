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
#save_results_to = 'F:/MOLI/result/'
save_results_to = 'F:/MOLI/result/res2/rna/'
#torch.manual_seed(42)
random.seed(42)

max_iter = 1

print('AB*************************************************************')
TCGAEx = pd.read_csv("brca_cnv_subtype.csv", 
                    index_col=0, header=None)

TCGAEx=TCGAEx.values
TCGAEx=TCGAEx[:,0:317]
#print(TCGAEx)

label=TCGAEx[0,:]
data=TCGAEx[1:,:]
#print(label)

label[np.where(label=='lminalA')]=1
label[np.where(label=='lminalB')]=0

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
    acctr1_all=[]
    accts1_all=[]
    acctr2_all=[]
    accts2_all=[]
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
            #marg = mrg
            lrE = lre
            epoch = epch

            costtr = []
            auctr = []
            costts = []
            aucts = []
            acctr1=[]
            accts1=[]
            acctr2=[]
            accts2=[]
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
                        nn.Linear(Z_in, 1),
                        nn.Dropout(rate),
                        nn.Sigmoid())
                def forward(self, x):
                    return self.FC(x)

            torch.cuda.manual_seed_all(42)

            AutoencoderE = AEE()


            solverE = optim.Adagrad(AutoencoderE.parameters(), lr=lrE)

          

            Clas = Classifier()
            SolverClass = optim.Adagrad(Clas.parameters(), lr=lrCL, weight_decay = wd)
            C_loss = torch.nn.BCELoss()
            print('epoch_all',epoch)
            for it in range(epoch):
                #print('epoch:',it)
                epoch_cost4 = []
                epoch_cost3 = []
                p_real=[]
                p_pred=[]
                n_real=[]
                n_pred=[]
                p_realt=[]
                p_predt=[]
                n_realt=[]
                n_predt=[]
               
                num_minibatches = int(n_sampE / mb_size) 
               
                for i, (dataE, target) in enumerate(trainLoader):
                    
                    flag = 0
                    AutoencoderE.train()

                    Clas.train()
                  
                    if torch.mean(target)!=0. and torch.mean(target)!=1.: 
                        ZEX = AutoencoderE(dataE)
                        Pred = Clas(ZEX)
                        

                      
                        loss=C_loss(Pred,target.view(-1,1))
                        y_true = target.view(-1,1)
                        y_pred = Pred

                        #AUC
                        AUC = roc_auc_score(y_true.detach().numpy(),y_pred.detach().numpy()) 
                        
                       
                        #acc
                        mask = y_pred.ge(0.5).float()
                        for i in range(len(y_true)):
                            if(y_true[i]==1):
                                p_real.append(y_true[:,0][i])
                                p_pred.append(mask[:,0][i])
                            else:
                                n_real.append(y_true[:,0][i])
                                n_pred.append(mask[:,0][i])

                        
                                
                        solverE.zero_grad()
                        SolverClass.zero_grad()

                        loss.backward()

                        solverE.step()
                        SolverClass.step()

                        epoch_cost4.append(loss)
                        epoch_cost3.append(AUC)
                flag = 1
                

                if flag == 1:
                    costtr.append(torch.mean(torch.FloatTensor(epoch_cost4)))
                    auctr.append(np.mean(epoch_cost3))
                    #print('Iter-{}; Total loss: {:.4}'.format(it, loss))
                    p_pred=torch.FloatTensor(p_pred)
                    p_real=torch.FloatTensor(p_real)
                    n_pred=torch.FloatTensor(n_pred)
                    n_real=torch.FloatTensor(n_real)
                    tp=(p_pred==p_real).sum().float()
                    fn=len(p_real)-tp
                    tn=(n_pred==n_real).sum().float()
                    fp=len(n_real)-tn
                    acc1=(tp+tn)/(len(p_real)+len(n_real))
                    sen=tp/len(p_real)
                    spe=tn/len(n_real)
                    acc2=(sen+spe)/2
                    acctr1.append(acc1)
                    acctr2.append(acc2)
                    sentr.append(sen)
                    spetr.append(spe)
             
                    
                with torch.no_grad():

                    AutoencoderE.eval()
                    Clas.eval()

                    ZET = AutoencoderE(TX_testE)
                    PredT = Clas(ZET)

                    
                    lossT=C_loss(PredT,ty_testE.view(-1,1))
                    y_truet = ty_testE.view(-1,1)
                    y_predt = PredT
                    AUCt = roc_auc_score(y_truet.detach().numpy(),y_predt.detach().numpy())

                    #acc
                    maskt = y_predt.ge(0.5).float()
                    for i in range(len(y_truet)):
                        if(y_truet[i]==1):
                            p_realt.append(y_truet[:,0][i])
                            p_predt.append(maskt[:,0][i])
                        else:
                            n_realt.append(y_truet[:,0][i])
                            n_predt.append(maskt[:,0][i])
                    p_predt=torch.FloatTensor(p_predt)
                    p_realt=torch.FloatTensor(p_realt)
                    n_predt=torch.FloatTensor(n_predt)
                    n_realt=torch.FloatTensor(n_realt)
                    tp=(p_predt==p_realt).sum().float()
                    fn=len(p_realt)-tp
                    tn=(n_predt==n_realt).sum().float()
                    fp=len(n_realt)-tn
                    acc1=(tp+tn)/len(y_truet)
                    sen=tp/len(p_realt)
                    spe=tn/len(n_realt)
                    acc2=(sen+spe)/2
                    accts1.append(acc1)
                    accts2.append(acc2)
                    sents.append(sen)
                    spets.append(spe)

                    costts.append(lossT)
                    aucts.append(AUCt)

                
                
                
                if(it==epoch-1):
                   
                    
                    print('acctest1:',acc1)
                    print('acctest2:',acc2)
                    print('sentest:',sen)
                    print('spetest:',spe)
                    
                    #ROC
                    fpr, tpr, thresholds = roc_curve(y_truet, y_predt)
                    mean_tpr += np.interp(mean_fpr, fpr, tpr)   # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
                    if(k%5==0):
                        temp_mean_tpr=mean_tpr/k
                        mean_tpr[0] = 0.0           # 将第一个真正例=0 以0为起点

                        #roc_auc = auc(fpr, tpr)  # 求auc面积

                        mean_auc = auc(mean_fpr, temp_mean_tpr)

         

                        plt.plot(mean_fpr, temp_mean_tpr, label='Mean ROC repeats {0:.1f} (area = {1:.2f})'.format(k//5,mean_auc), lw=1)

                        #plt.plot(fpr, tpr, lw=1, label='ROC fold {0:.2f} (area = {1:.2f})'.format(k, roc_auc))    # 画出当前分割数据的ROC曲线
            
                    


            costtr_all.append(costtr)
            auctr_all.append(auctr)
            costts_all.append(costts)
            aucts_all.append(aucts)
            acctr1_all.append(acctr1)
            accts1_all.append(accts1)
            acctr2_all.append(acctr2)
            accts2_all.append(accts2)
            sentr_all.append(sentr)
            sents_all.append(sents)
            spetr_all.append(spetr)
            spets_all.append(spets)
            
            plt.plot(np.squeeze(costtr), '-r',np.squeeze(costts), '-b')
            plt.ylabel('Total cost')
            plt.xlabel('iterations (per tens)')

            title = 'Cost  iter = {}, fold = {}, mb_size = {},  h_dim = {} , lrE = {}, epoch = {}, rate = {}, wd = {}, lrCL = {}'.\
                          format(iters, k, mbs, hdm,  lre, epch, rate, wd, lrCL)

            #title = 'Cost  iter = {}, fold = {}, mb_size = {},  h_dim = {}, marg = {}, lrE = {}, epoch = {}, rate = {}, wd = {}, lrCL = {}, lam = {}'.\
            #              format(iters, k, mbs, hdm, mrg, lre, epch, rate, wd, lrCL, lam)

            plt.suptitle(title)
            plt.savefig(save_results_to + title + '.png', dpi = 150)
            plt.close()

            plt.plot(np.squeeze(auctr), '-r',np.squeeze(aucts), '-b')
            plt.ylabel('AUC')
            plt.xlabel('iterations (per tens)')

            title = 'AUC  iter = {}, fold = {}, mb_size = {},  h_dim = {},  lrE = {}, epoch = {}, rate = {}, wd = {}, lrCL = {}, '.\
                          format(iters, k, mbs, hdm, lre, epch, rate, wd, lrCL)        

            plt.suptitle(title)
            plt.savefig(save_results_to + title + '.png', dpi = 150)
            plt.close()
            
            
            if k%50==0:
                
                #ROC-mean
                plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck') # 画对角线
                mean_tpr /= k   # 求数组的平均值

                mean_tpr[-1] = 1.0   # 坐标最后一个点为（1,1）  以1为终点

                mean_auc = auc(mean_fpr, mean_tpr)

         

                plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = {0:.2f})'.format(mean_auc), lw=2)

         

                plt.xlim([-0.05, 1.05])     # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体

                plt.ylim([-0.05, 1.05])

                plt.xlabel('False Positive Rate')

                plt.ylabel('True Positive Rate')    # 可以使用中文，但需要导入一些库即字体

                plt.title('Receiver operating characteristic example')

                plt.legend(loc="lower right")
                title='cnv-ROC(A=1,B=0)-skf10 '
                plt.suptitle(title)
                plt.savefig(save_results_to + title + '.png', dpi = 150)
                plt.close()

                
                costtr_all=np.array(costtr_all)
                auctr_all=np.array(auctr_all)
                costts_all=np.array(costts_all)
                aucts_all=np.array(aucts_all)
                acctr1_all=np.array(acctr1_all)
                accts1_all=np.array(accts1_all)

                acctr2_all=np.array(acctr2_all)
                accts2_all=np.array(accts2_all)
                sentr_all=np.array(sentr_all)
                sents_all=np.array(sents_all)
                spetr_all=np.array(spetr_all)
                spets_all=np.array(spets_all)
                #ctr=sum(costtr_all)/5
                #atr=sum(auctr_all)/5
                #cts=sum(costts_all)/5
                #ats=sum(aucts_all)/5
                costtr5=sum(costtr_all)/k
                costts5=sum(costts_all)/k
                auctr5=sum(auctr_all)/k
                aucts5=sum(aucts_all)/k
                acctr15=sum(acctr1_all)/k
                accts15=sum(accts1_all)/k

                acctr25=sum(acctr2_all)/k
                accts25=sum(accts2_all)/k
                sentr5=sum(sentr_all)/k
                sents5=sum(sents_all)/k
                spetr5=sum(spetr_all)/k
                spets5=sum(spets_all)/k
                print('acc1-200:',accts15[-1])
                print('acc1max:',max(accts15))

                print('acc2-200:',accts25[-1])
                print('acc2max:',max(accts25))
                print('cost200:',costts5[-1])
                print('costmin:',min(costts5))
                print('auc200:',aucts5[-1])
                print('aucmax:',max(aucts5))
                print('acc1:',accts15)
                print('acc2:',accts25)
                print('cost:',costts5)
                print('auc:',aucts5)
                #ctr=ctr/5
                #atr=atr/5
                #cts=cts/5
                #ats=ats/5
                
                plt.plot(np.squeeze(costtr5), '-r',np.squeeze(costts5), '-b')
                plt.ylabel('Total cost')
                plt.xlabel('epoch')

                title = 'cnv_ Cost(A=1,B=0)-skf10   mb_size = {},  h_dim = {} , lrE = {}, epoch = {}, rate = {}, wd = {}, lrCL = {}'.\
                              format( mbs, hdm,  lre, epch, rate, wd, lrCL)

                #title = 'Cost  iter = {}, fold = {}, mb_size = {},  h_dim = {}, marg = {}, lrE = {}, epoch = {}, rate = {}, wd = {}, lrCL = {}, lam = {}'.\
                #              format(iters, k, mbs, hdm, mrg, lre, epch, rate, wd, lrCL, lam)

                plt.suptitle(title)
                plt.savefig(save_results_to + title + '.png', dpi = 150)
                plt.close()

                plt.plot(np.squeeze(auctr5), '-r',np.squeeze(aucts5), '-b')
                plt.ylabel('AUC')
                plt.xlabel('epoch')

                title = 'cnv_ AUC(A=1,B=0)-skf10  mb_size = {},  h_dim = {},  lrE = {}, epoch = {}, rate = {}, wd = {}, lrCL = {}, '.\
                              format( mbs, hdm, lre, epch, rate, wd, lrCL)        

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


                plt.plot(np.squeeze(sentr5), '-r',np.squeeze(sents5), '-b')
                plt.ylabel('sensitivity')
                plt.xlabel('epoch')

                title = 'cnv_ sensitivity(A=1,B=0)-skf10   mb_size = {},  h_dim = {},  lrE = {}, epoch = {}, rate = {}, wd = {}, lrCL = {}, '.\
                              format( mbs, hdm, lre, epch, rate, wd, lrCL)        

                plt.suptitle(title)
                plt.savefig(save_results_to + title + '.png', dpi = 150)
                plt.close()



                plt.plot(np.squeeze(spetr5), '-r',np.squeeze(spets5), '-b')
                plt.ylabel('specificity')
                plt.xlabel('epoch')

                title = 'cnv_ specificity(A=1,B=0)-skf10  mb_size = {},  h_dim = {},  lrE = {}, epoch = {}, rate = {}, wd = {}, lrCL = {}, '.\
                              format( mbs, hdm, lre, epch, rate, wd, lrCL)        

                plt.suptitle(title)
                plt.savefig(save_results_to + title + '.png', dpi = 150)
                plt.close()

                




##----------------------------------
##  Code Version 1.0
##  This breast cancer code for logistic regression with Elastic net penalty embeded in the concatenation-based framework.
##  Concate_EN
##  Created by Zi-Yi Yang 
##  Modified by Zi-Yi Yang on July 23, 2019
##  Concact: yangziyi091100@163.com
##----------------------------------

library(glmnet)
library(caret)
library(plyr)
library(randomForest)
library(ROCR)
library(ROSE)
source("F:/MOLI/gdac/MSPL-master/MSPL-master/3-Breast/Function_performance.R")

CVgroup <- function(k,datasize,seed){
  cvlist <- list()
  set.seed(seed)
  n <- rep(1:k,ceiling(datasize/k))[1:datasize]    #将数据分成K份，并生成的完成数据集n
  temp <- sample(n,datasize)   #把n打乱
  x <- 1:k
  dataseq <- 1:datasize
  cvlist <- lapply(x,function(x) dataseq[temp==x])  #dataseq中随机生成k个随机有序数据列
  return(cvlist)
}

##--------------
# 1. Load data
##--------------
#load("D:/Ziyi/School/PMO/12.Multi_omics/7.MSPL-code/3-Breast/1-data/1-datatrainTestDatasetsNormalized.RDATA")
options("expressions"=500000)
memory.limit(size=80000000)
data_cnv<-read.csv("F:/MOLI/data/brca_cnv_subtype_trans.csv",header=T,row.names=1)
data_meth<-read.csv("F:/MOLI/data/brca_meth_subtype_trans.csv",header=T,row.names=1)
data_rna<-read.csv("F:/MOLI/data/brca_rna_subtype_trans.csv",header=T,row.names=1)

datadf1=as.data.frame(rbind(as.matrix(data_cnv[1:361,]),as.matrix(data_cnv[362:606,])))
datadf2=as.data.frame(rbind(as.matrix(data_meth[1:361,]),as.matrix(data_meth[362:606,])))
datadf3=as.data.frame(rbind(as.matrix(data_rna[1:361,]),as.matrix(data_rna[362:606,])))

datadf1$lab<-as.factor(datadf1$lab)
datadf2$lab<-as.factor(datadf2$lab)
datadf3$lab<-as.factor(datadf3$lab)
datadf1=as.matrix(datadf1)
datadf2=as.matrix(datadf2)
datadf3=as.matrix(datadf3)

label=rbind(as.matrix(data_cnv[1:361,1]),as.matrix(data_cnv[362:606,1]))

## Preparing data

#train_data<-data[,c(2:485)]
#train_label<-train_data[1,]
#train_data<-train_data[2:42667,]

#test_data<-data[,c(486:607)]
#test_label<-test_data[1,]
#test_data<-test_data[2:42667,]


##--------------
# 2. Concatenation
##--------------
#combined_train_data <- t(train_data)
#combined_test_data <- t(test_data)



## Normalize entire dataset
#lib.size <- colSums(combined_train_data)

#combined_train_data <- log2(t(combined_train_data+0.5)/(lib.size+1) * 1e+06)

#lib.size <- colSums(combined_test_data)

#combined_test_data <- log2(t(combined_test_data+0.5)/(lib.size+1) * 1e+06)

#train_label<-t(train_label)
#test_label<-t(test_label)


k<-5
datasize<-nrow(datadf1)
print(datasize)
cvlist<-CVgroup(k = k,datasize = datasize,seed = 100)
#print(cvlist)
for(jj in 1:k){
    train_data_meth <- datadf2[-cvlist[[jj]],]  #刚才通过cvgroup生成的函数
	train_data_rna <- datadf3[-cvlist[[jj]],]
	train_data_cnv <- datadf1[-cvlist[[jj]],]
    test_data_meth <- datadf2[cvlist[[jj]],]
	test_data_rna <- datadf3[cvlist[[jj]],]
	test_data_cnv <- datadf1[cvlist[[jj]],]
	train_label<-label[-cvlist[[jj]],]
	test_label<-label[cvlist[[jj]]]
	
	#data_balanced_both1 <- ovun.sample(lab ~ ., data = train_data_cnv, method = "both", p=0.5, N=289, seed = 1)$data
	#print(table(data_balanced_both1$lab))
	#data_balanced_both2 <- ovun.sample(lab ~ ., data = train_data_meth, method = "both", p=0.5, N=289, seed = 1)$data
	#print(table(data_balanced_both2$lab))
	#data_balanced_both3 <- ovun.sample(lab ~ ., data = train_data_rna, method = "both", p=0.5, N=289, seed = 1)$data
	#print(table(data_balanced_both3$lab))
	
	train_data <- list("meth" = as.matrix(train_data_meth[,2:14286]),
                   "rna" = as.matrix(train_data_rna[,2:13196]),
                   "cnv" = as.matrix(train_data_cnv[,2:15187]))
				   
	#train_label<-train_data_cnv[,1]
	colnames(train_data$meth) <- paste("meth", colnames(train_data$meth), sep = "_")
	colnames(train_data$rna) <- paste("rna", colnames(train_data$rna), sep = "_")
	colnames(train_data$cnv) <- paste("cnv", colnames(train_data$cnv), sep = "_")
				   
	test_data <- list("meth" = as.matrix(test_data_meth[,2:14286]),
                   "rna" = as.matrix(test_data_rna[,2:13196]),
                   "cnv" = as.matrix(test_data_cnv[,2:15187]))

	colnames(test_data$meth) <- paste("meth", colnames(test_data$meth), sep = "_")
	colnames(test_data$rna) <- paste("rna", colnames(test_data$rna), sep = "_")
	colnames(test_data$cnv) <- paste("cnv", colnames(test_data$cnv), sep = "_")
	print('111')
    # 2.EnsembleRF
	##--------------
	ensemblePanels <- lapply(train_data, function(i){
	  RF.colon <- randomForest(x = i, y = factor(train_label), importance = TRUE, ntree = 8)
	})
	print('222')
	ensembleValiPredction <- mapply(function(fit,x){
	  valprediction <- predict(fit,x)
	}, fit = ensemblePanels, x = train_data)
	print('333')
	ensembleTestPredction <- mapply(function(fit,x){
	  tsprediction <- predict(fit,x)
	}, fit = ensemblePanels, x = test_data)
	print('444')
	valprediction <- evaluate.ensemble(ensembleValiPredction, train_label)
	tsprediction <- evaluate.ensemble(ensembleTestPredction, test_label)

	#ensembleCoef.idx <- list()
	#ensembleCoef.name <- list()

	#for(i in 1:length(ensemblePanels)){
	#  ensembleCoef.idx[[i]] <- which(ensemblePanels[[i]]$importance[,4]!=0)
	#  ensembleCoef.name[[i]] <- colnames(train_data[[i]])[ensembleCoef.idx[[i]]]
	#}

	##----------------------------
	# 3. Evaluate the performance
	##----------------------------

	#View(train_label)
	#View(valprediction)
	#View(test_label)
	#View(tsprediction)
	
	conMatrix.train <- confusionMatrix(as.factor(valprediction),as.factor(train_label))
	conMatrix.test <- confusionMatrix(as.factor(tsprediction),as.factor(test_label),mode = "prec_recall")

	#pred <- prediction(tsprediction, test_label)
	#auc <- performance(pred,'auc')
	#auc=unlist(slot(auc,"y.values"))
	perf.ConcatenationEN <- list("Perf.Train" = conMatrix.train,
                             "Perf.Test" = conMatrix.test)
							 #"AUC" = auc)

	#View(perf.ConcatenationEN)
	cat('********************************************************************************\n')
	cat('k=',jj,"\n")
	print(perf.ConcatenationEN)
	cat("\n")
	}
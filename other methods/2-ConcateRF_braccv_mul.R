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
#data<-read.csv("F:/MOLI/data/brca_all_606.csv",header=FALSE)


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

label=rbind(as.matrix(data_cnv[1:361,1]),as.matrix(data_cnv[362:606,1]))
## Preparing data

#train_data<-data[,c(2:485)]
#train_label<-train_data[1,]
#train_data<-train_data[2:42667,]

#test_data<-data[,c(486:607)]
#test_label<-test_data[1,]
#test_data<-test_data[2:42667,]

##data<-data[,c(2:362)]
##label<-data[1,]
##data<-data[2:42667,]

##--------------
# 2. Concatenation
##--------------

#combined_train_data <- t(train_data)
#combined_test_data <- t(test_data)


#data<-t(data)
#label<-t(label)
#View(label)
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

for(i in 1:k){
    #print(cvlist[[i]])
	test_label<-label[cvlist[[i]]]
    #combined_train_data <- data[-cvlist[[i]],]  #刚才通过cvgroup生成的函数
    #combined_test_data <- data[cvlist[[i]],]
	combined_train_data1 <- datadf1[-cvlist[[i]],]  #刚才通过cvgroup生成的函数
    combined_test_data1 <- datadf1[cvlist[[i]],]
	combined_train_data2 <- datadf2[-cvlist[[i]],]  #刚才通过cvgroup生成的函数
    combined_test_data2 <- datadf2[cvlist[[i]],]
	combined_train_data3 <- datadf3[-cvlist[[i]],]  #刚才通过cvgroup生成的函数
    combined_test_data3 <- datadf3[cvlist[[i]],]
	#m<- nrow(combined_train_data1)
	#data_balanced_both1 <- ovun.sample(lab ~ ., data = combined_train_data1, method = "both", p=0.5, N=256, seed = 1)$data
	#print(table(data_balanced_both1$lab))
	#data_balanced_both2 <- ovun.sample(lab ~ ., data = combined_train_data2, method = "both", p=0.5, N=256, seed = 1)$data
	#print(table(data_balanced_both2$lab))
	#data_balanced_both3 <- ovun.sample(lab ~ ., data = combined_train_data3, method = "both", p=0.5, N=256, seed = 1)$data
	#print(table(data_balanced_both3$lab))
	
	##combined_train_data<- cbind(data_balanced_both1[,c(2:15187)],data_balanced_both2[,c(2:14286)])
	##combined_train_data<- cbind(combined_train_data,data_balanced_both3[,c(2:13196)])
	##combined_test_data<- cbind(combined_test_data1[,c(2:15187)],combined_test_data2[,c(2:14286)])
	##combined_test_data<- cbind(combined_test_data,combined_test_data3[,c(2:13196)])
	
	combined_train_data<- cbind(combined_train_data1[,c(2:15187)],combined_train_data2[,c(2:14286)])
	combined_train_data<- cbind(combined_train_data,combined_train_data3[,c(2:13196)])
	combined_test_data<- cbind(combined_test_data1[,c(2:15187)],combined_test_data2[,c(2:14286)])
	combined_test_data<- cbind(combined_test_data,combined_test_data3[,c(2:13196)])
	
	#train_label<-label[-cvlist[[i]],]
	#test_label<-label[cvlist[[i]],]
	
	#train_label<-combined_train_data[,1]
	
	#print(cvlist[[i]])
	#test_label<-label[cvlist[[i]],]
	
	#combined_train_data<- as.matrix(combined_train_data)
	#combined_test_data<- as.matrix(combined_test_data)
	combined_train_data<- as.matrix(combined_train_data)
	combined_test_data<- as.matrix(combined_test_data)
	train_label<-combined_train_data1[,1]
	print(dim(combined_train_data))
	print(dim(combined_test_data))
	print(dim(as.matrix(train_label)))
	print(dim(as.matrix(test_label)))
	print('111')
    net <- randomForest(x = combined_train_data, y = factor(train_label), 
                    importance = TRUE, ntree = 10)
						
	print('222')
    valprediction <- predict(net, combined_train_data)
	print('333')
	tsprediction <- predict(net, combined_test_data)
	#coef.idx <- which(net$importance[,4]!=0)
	#coef.name <- colnames(combined_train_data)[coef.idx]
	#View(tsprediction)
    #valprediction <- as.numeric(valprediction)
	#tsprediction <- as.numeric(tsprediction)
	valprediction <- as.numeric(as.character(valprediction))
	tsprediction <- as.numeric(as.character(tsprediction))
	#coefprediction <- predict(cvfit,type="coefficients",s="lambda.min")  # coef
	#coef.idx <- which(coefprediction$`1`!=0)[-1]
	#coef.name <- rownames(coefprediction$`1`[-1])[coef.idx]

##----------------------------
# 3. Evaluate the performance
##----------------------------

	#View(train_label)
	#View(valprediction)
	
	#View(test_label[1])
	#View(tsprediction)
	conMatrix.train <- confusionMatrix(as.factor(valprediction),as.factor(train_label))
	conMatrix.test <- confusionMatrix(as.factor(tsprediction),as.factor(test_label),mode = "prec_recall")
	#pred <- prediction(tsprediction, test_label)
	#auc <- performance(pred,'auc')
	#auc=unlist(slot(auc,"y.values"))
	perf.ConcatenationEN <- list(#"Feature" = coef.name,
                             #"Feature.idx" = coef.idx,
                             "Perf.Train" = conMatrix.train,
                             "Perf.Test" = conMatrix.test)
							 #"AUC" = auc)

	#View(perf.ConcatenationEN)
	cat('******************************************\n')
	cat('k=',i,"\n")
	print(perf.ConcatenationEN)
	cat("\n")
	}
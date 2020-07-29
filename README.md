# DeepMO
DeepMO:Classifying Breast Cancer Subtypes Using Deep Neural Networks Based on Multi-omics Data

This repository provides some codes for this paper.

The repository can be divided into three parts:
1. Codes for binary classification.
2. Codes for multiple classificaion.
3. Codes for other state-of-art methods of omics data integration.

## DeepMO model
We take multiple classification using multi-omics as example case.

The raw data of breast cancer multi-omics can be download from The Cancer Genome Atlas (TCGA). Here, we just use part of data as example, which is provided in example data folder. The workflow of this code is described as follows:

(1) Read corresponding omics data file.<br>
(2) Preprocess data, including data normalization and feature selection. The number of selected features is adjustable.<br>
(3) Set the hyper-parameters of deep neural network, such as the number of nodes in the hidden layers, learning rates, mini-batch size, weight decay, the dropout rate and the number of epochs.<br>
(4) Divide dataset into train dataset and test dataset.<br>
(5) Define and train deep neural network and test on test dataset.<br>
(6) Show the output.

In this study, we repeat the process for 10 times and use the mean of results to reduce error.All the codes in this section are accomplished in Python3.7.

## Other compared methods
To further evaluate the performance of DeepMO on multiple classification, we selected some state-of-art methods of omics data integration, including the logistic regression model/multinational model with Elastic Net (EN) regularization and Random Forest (RF) in the concatenation and ensemble frameworks. The concrete implementation can be found from [1]. All the codes in this section is implemented in R 3.6.2.

## Reference
[1]Yang Z Y , Xia L Y , Zhang H , et al. MSPL: Multimodal Self-Paced Learning for multi-omics feature selection and data integration[J]. IEEE Access, 2019, PP(99):1-1.

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix

from sklearn.preprocessing import normalize
from sklearn.preprocessing import binarize

import sklearn.metrics as metrics

import networkx as nx

localDataDir = "/Users/sagarsetru/Documents/Princeton/cos424/hw3/Assignment3_Bitcoin/data/"
undirectedDataDir = localDataDir+"featuresFromUndirectedGraph/"
directedDataDir = localDataDir+"featuresFromDirectedGraph/"

def auc( true, pred ):
    '''plot a ROC curve from true and predicted binary input vectors'''
    fpr, tpr, thresholds = metrics.roc_curve( true, pred )
    roc_auc = metrics.roc_auc_score( true, pred )
    return roc_auc

def loadUndirectedGraphFeatures(PathToFile,FileName):
    '''Load undirected graph features, include path and filename as strings, include "/" at end of PathToFile'''
    '''returns matrix with columns: sender, receiver, feature value'''
    '''Last column in this matrix is the feature vector'''
    # undirected graph features
    #adamic adar index
    featureMatrix = np.loadtxt(PathToFile+FileName,usecols=(2,))
    return featureMatrix
    #preferential attachment
    #pa = np.loadtxt(undirectedDataDir+'preferential_attachment.txt',usecols=(2,))
    #resource allocation
    #ra = np.loadtxt(undirectedDataDir+'resource_allocation.txt',usecols=(2,))
    #jaccard coefficient
    #jc = np.loadtxt(undirectedDataDir+'jaccard_coefficient.txt',usecols=(2,))

def loadDirectedGraphFeatures(PathToFile,FileName,senderIDs,receiverIDs):
    '''load directed features, include path and filename as strings, include "/" at end of PathToFile, must also include the senderIDs and receiverIDs from testing data set'''
    '''returns matrix with 4 columns: sender, receiver, sender feature value, receiver feature value'''
    '''last 2 columnds in this matrix are the feature vectors'''
    # directed graph features
    feature = np.loadtxt(PathToFile+FileName,delimiter=',',usecols=(1,))
    featureSenders = feature[senderIDs]
    featureReceivers = feature[receiverIDs]
    featureMatrix = np.vstack((featureSenders,featureReceivers))
    return featureMatrix.T

def plotROC( true, pred ):
    '''plot a ROC curve from true and predicted binary input vectors'''
    fpr, tpr, thresholds = metrics.roc_curve( true, pred )
    roc_auc = metrics.roc_auc_score( true, pred )
    plt.style.use('ggplot')
    plt.figure(num=1,figsize=(10, 8))
    plt.subplot(111)
    plt.plot(fpr, tpr, color='magenta', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    return plt

te = pd.read_csv(localDataDir+'testTriplets.txt', header=None, delimiter=r"\s+")
te_s = te[0].values
te_r = te[1].values
te_b = te[2].values


a = loadDirectedGraphFeatures(directedDataDir,'authority.csv',te_s,te_r)
h = loadDirectedGraphFeatures(directedDataDir,'hub.csv',te_s,te_r)
pr =  loadDirectedGraphFeatures(directedDataDir,'pagerank.csv',te_s,te_r)

aai = loadUndirectedGraphFeatures(undirectedDataDir,'adamic_adar_index.txt')
pa = loadUndirectedGraphFeatures(undirectedDataDir,'preferential_attachment.txt')
jc = loadUndirectedGraphFeatures(undirectedDataDir,'jaccard_coefficient.txt')
ra = loadUndirectedGraphFeatures(undirectedDataDir,'resource_allocation.txt')

all_data = np.vstack((te_s,te_r,aai,pa,jc,ra,a[:,0],a[:,1],h[:,0],h[:,1],pr[:,0],pr[:,1]))
#np.savetxt('allGraphicalData.txt',all_data.T)

testMetricName = 'adamic_adar_index.txt'
testMetric = loadUndirectedGraphFeatures(undirectedDataDir,testMetricName)
thresholds = np.unique(np.sort(testMetric))
# blank array for AUCs
aucs = np.zeros(thresholds.shape)
rm = np.zeros(thresholds.shape)
counter = -1
for threshold in thresholds:
    counter += 1
    testMetric = loadUndirectedGraphFeatures(undirectedDataDir,testMetricName)
    #s_r = s.multiply(r)
    #mle_b = s_r.sum(axis=1)
    #mle_b = np.asarray(mle_b)
    #mle_b = mle_b[:,0]
    #mle_b = mle_2
    testMetric[np.where(testMetric <= threshold)] = 0
    testMetric[np.where(testMetric > threshold)] = 1
    print np.sum(testMetric)
    aucs[counter] = auc(te_b,testMetric)
    rm[counter]=metrics.zero_one_loss(te_b,testMetric)
    #aucs[counter] = auc(te_b,np.where(np.logical_and(mle_2 < threshold, mle_2 >= threshold), 0, 1))
#...

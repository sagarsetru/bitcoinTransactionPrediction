import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix

from sklearn.preprocessing import normalize
from sklearn.preprocessing import binarize

import sklearn.metrics as metrics

import networkx as nx

def loadGeneratorToMatrix ( m_shape, g  ):
    counter = -1
    m = np.zeros(m_shape)
    for u, v, p in g:
        counter += 1
        m[counter,:]=[u,v,p]
    #...
    return m

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

def auc( true, pred ):
    '''plot a ROC curve from true and predicted binary input vectors'''
    fpr, tpr, thresholds = metrics.roc_curve( true, pred )
    roc_auc = metrics.roc_auc_score( true, pred )
    return roc_auc

def firstOrderMLE(tr_A, te_s, te_r):
    mle = tr_A[te_s,te_r]
    mle = np.asarray(mle.T)
    mle = mle[:,0]
    return mle

def secondOrderMLE(tr_A, te_s, te_r):
    #mle_2 = np.zeros(te_s.shape)
    s = tr_A[te_s,:]
    r = tr_A[:,te_r].T
    s_r = s.multiply(r)
    mle_2 = s_r.sum(axis=1)
    mle_2 = np.asarray(mle_2)
    mle_2 = mle_2[:,0]
    return mle_2
    
# import training and testing data
baseDirData = '/Users/sagarsetru/Documents/Princeton/cos424/hw3/Assignment3_Bitcoin/data/'

tr = pd.read_csv(baseDirData+'txTripletsCounts.txt', header=None, delimiter=r"\s+")
te = pd.read_csv(baseDirData+'testTriplets.txt', header=None, delimiter=r"\s+")

# get senders, receivers, number of transactions in training data
tr_s = tr[0].values
tr_r = tr[1].values
tr_nt = tr[2].values

# get testing senders, receivers, binary transaction score
te_s = te[0].values
te_r = te[1].values
te_b = te[2].values

# fill sparse matrix
tr_A = csr_matrix((tr_nt, (tr_s, tr_r)),shape=(np.max(tr_s)+1, np.max(tr_s)+1))
te_A = csr_matrix((te_b, (te_s, te_r)),shape=(np.max(te_s)+1, np.max(te_s)+1))

# normalize tr_A
#http://stackoverflow.com/questions/8904694/how-to-normalize-a-2-dimensional-numpy-array-in-python-less-verbose
#tr_A = normalize(tr_A, axis = 1, norm = 'l1')

### graph based methods
doGraphAnalysis = 1

makeDiGraph = 1
doPagerank = 0
doHits = 0


makeUnDiGraph = 0
doLinkPred = 0

nodePairs = np.rec.fromarrays([te_s, te_r])

if doGraphAnalysis == 1:
    if makeDiGraph == 1:
        print 'making directed graph...'
        G = nx.DiGraph(tr_A)
    #...

    if doPagerank == 1:
        print 'Calculating pagerank...'
        # get pagerank
        pr = nx.pagerank(G)

        writer = csv.writer(open('pagerank.csv', 'wb'))
        for key,value in pr.items():
            writer.writerow([key, value])
        #...
    #...

    # get hubs and authorities
    if doHits == 1:
        print 'Calculating hubs and authorities...'

        h,a = nx.hits(G)

        writer = csv.writer(open('hub.csv', 'wb'))
        for key,value in h.items():
            writer.writerow([key, value])
        #...

        writer = csv.writer(open('authority.csv', 'wb'))
        for key,value in a.items():
           writer.writerow([key, value])
        #...
    #...

    if makeUnDiGraph == 1:
        print 'making undirected graph...'
        G = nx.Graph(tr_A)
    #...
    
    if doLinkPred == 1:
        pairValsShape = te.shape
        
        print 'Calculating resource_allocation_index...'
        pairGen = nx.resource_allocation_index(G,nodePairs)
        pairVals = loadGeneratorToMatrix(pairValsShape,pairGen)
        np.savetxt('resource_allocation.txt',pairVals)
        '''
        print 'Calculating ra_index_soundarajan_hopcroft...'
        pairGen = nx.ra_index_soundarajan_hopcroft(G,nodePairs,community='community')
        pairVals = loadGeneratorToMatrix(te,pairGen)
        np.savetxt('ra_index_soundarajan_hopcroft.txt',pairVals)
        
        print 'Calculating cn_soundarajan_hopcroft...'
        pairGen = nx.cn_soundarajan_hopcroft(G,nodePairs)
        pairVals = loadGeneratorToMatrix(tpairValsShape,pairGen)
        np.savetxt('cn_soundarajan_hopcroft.txt',pairVals)
        '''
        print 'Calculating preferential_attachment...'
        pairGen = nx.preferential_attachment(G,nodePairs)
        pairVals = loadGeneratorToMatrix(pairValsShape,pairGen)
        np.savetxt('preferential_attachment.txt',pairVals)
        
        print 'Calculating adamic_adar_index...'
        pairGen = nx.adamic_adar_index(G,nodePairs)
        pairVals = loadGeneratorToMatrix(pairValsShape,pairGen)
        np.savetxt('adamic_adar_index.txt',pairVals)
        
        print 'Calculating jaccard_coefficient...'
        pairGen = nx.jaccard_coefficient(G,nodePairs)
        pairVals = loadGeneratorToMatrix(pairValsShape,pairGen)
        np.savetxt('jaccard_coefficient.txt',pairVals)
    #...
#...

saveGraphML = 0
if saveGraphML == 1:
    pos = nx.spring_layout(G)

    for node,(x,y) in pos.items():
        G.node[node]['x'] = float(x)
        G.node[node]['y'] = float(y)
    #...
    nx.write_graphml(G,'directed_graph.graphml')


doStationaryMLE=0
if doStationaryMLE == 1:
    # get MLE for 1st order transition matrix
    mle = firstOrderMLE(tr_A,te_s,te_r)
    # get MLE for 2nd order transition matrix
    #mle_2 = secondOrderMLE(tr_A,te_s,te_r)
    s = tr_A[te_s,:]
    r = tr_A[:,te_r].T
    s_r = s.multiply(r)
    mle_2 = s_r.sum(axis=1)
    mle_2 = np.asarray(mle_2)
    mle_2 = mle_2[:,0]

    #test=np.where(np.logical_and(mle_2 < mle_2[1], mle_2 >= mle_2[1]), 0, 1)

    # get all unique MLEs
    thresholds = np.unique(np.sort(mle_2))
    # blank array for AUCs
    aucs = np.zeros(thresholds.shape)
    rm = np.zeros(thresholds.shape)
    counter = -1
    for threshold in thresholds:
        counter += 1
        s_r = s.multiply(r)
        mle_b = s_r.sum(axis=1)
        mle_b = np.asarray(mle_b)
        mle_b = mle_b[:,0]
        #mle_b = mle_2
        mle_b[np.where(mle_b <= threshold)] = 0
        mle_b[np.where(mle_b > threshold)] = 1
        print np.sum(mle_b)
        aucs[counter] = auc(te_b,mle_b)
        rm[counter]=metrics.zero_one_loss(te_b,mle_b)
        #aucs[counter] = auc(te_b,np.where(np.logical_and(mle_2 < threshold, mle_2 >= threshold), 0, 1))
    #...
    np.savetxt('probThresholds_2ndOrder.txt', threshold)
    np.savetxt('AUCs_2ndOrder.txt', aucs)
    np.savetxt('rm_2ndOrder.txt', rm)
#...

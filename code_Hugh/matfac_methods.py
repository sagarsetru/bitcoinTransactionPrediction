#!/usr/bin/python

"""
Script for COS424 Assignment 3: Bitcoin
Input:
    Number of transactions between (sender,receiver) pairs
    from a set of addresses
Processing:
    Produce a set of latent variables using several dimension reduction methods
    Fit a (two component) Gaussian mixture model on the training data - EM
    Predict the probability that each pair in the test set belongs to each
        of the clusters - gives a preciction and uncertainty.
Evaluation:
    Use ROC curves to compare the predicted and true results for the test set.
"""

import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import sklearn.decomposition as decomp
import sklearn.mixture as mix

def main():
    ## DATA IMPORT
    baseDirData = '/Users/hugh/Google Drive/Assignment3_Bitcoin/data/'

    tr = pd.read_csv(baseDirData+'txTripletsCounts.txt', header=None, \
                     delimiter=r"\s+", names = ['s','r','t'])
    te = pd.read_csv(baseDirData+'testTriplets.txt', header=None, \
                     delimiter=r"\s+", names = ['s','r','t'])

    ## SENDER ROWS
    # get senders, receivers, number of transactions in training data
    tr_s = tr['s'].values
    tr_r = tr['r'].values
    tr_nt = tr['t'].values
    # get testing senders, receivers, binary transaction score
    te_s = te['s'].values
    te_r = te['r'].values
    te_b = te['t'].values

    trn_spr = csr_matrix((tr_nt, (tr_s, tr_r)),shape=(np.max(tr_s)+1, np.max(tr_s)+1))
    test_spr = csr_matrix((te_b, (te_s, te_r)),shape=(np.max(te_s)+1, np.max(te_s)+1))


    ## RECEIVER ROWS
    rtr = tr.sort_values(by=['r','s'])
    rtr = rtr[['r','s','t']]
    rtrn_spr = csr_matrix((rtr['t'].values, (rtr['r'].values, rtr['s'].values)), \
                          shape=(np.max(rtr['s'].values)+1, np.max(rtr['s'].values)+1))



    ## DIMENSION REDUCTION
    # Produce a low dimensional version of the sender ordered training data
    tsvd = decomp.TruncatedSVD()    # n_components, random_state
    tsvd.fit(trn_spr)
    trn_low = tsvd.transform(trn_spr)

    # Produce a low dimensional version of the receiver ordered training data
    rtsvd = decomp.TruncatedSVD()
    rtsvd.fit(rtrn_spr)
    rtrn_low = rtsvd.transform(rtrn_spr)

    # Produce the training matrix - latent values for each s,r pair
    trn_pairs_matrix = np.concatenate((trn_low[tr_s,:],rtrn_low[tr_r,:]),axis=1)
    trn_pairs = pd.DataFrame(trn_pairs_matrix)


    ## MODEL FITTING
    # Fit a Gaussian Mixture Model to the low dimensional training data
    GMM = mix.GMM(n_components=2)
    GMM.fit(trn_low)


    ## PREDICTIONS

    # Extensions
    # Produce a reduced representation of each sender
    # Produce a reduced representation of each receiver
    # Produce an array - rows corresponding to s,r pairs with sender features,
    # then receiver features along the row.
    # Train the GMM on the s,r pairs we have
    # Predict for the s,r pairs we don't
    # Assess the missing data assumptions I have made





if __name__ == '__main__':
    main()

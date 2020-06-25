#Sagar Setru April 1 2016

import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix

from sklearn.preprocessing import normalize

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

# load into sparse matrix
tr_A = csr_matrix((tr_nt, (tr_s, tr_r)),shape=(np.max(tr_s)+1, np.max(tr_s)+1))
te_A = csr_matrix((te_b, (te_s, te_r)),shape=(np.max(te_s)+1, np.max(te_s)+1))

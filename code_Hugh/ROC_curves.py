#!/usr/bin/python

"""
Script for COS424 Assignment 3: Bitcoin
Calculates a ROC curve given two binary vectors indicating
the true and predicted outcomes of a series of trials.
"""

import sklearn.metrics as metrics
import matplotlib.pyplot as plt

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


def main():
    true_results = [ 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0 ]
    pred_results = [ 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0 ]
    plt = plotROC( true_results, pred_results )
    plt.show()

if __name__ == '__main__':
    main()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
import pickle
import seaborn as sns
import imp
imp.reload(plt); imp.reload(sns)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
import scipy.stats
from scipy import stats


'''
Helpers for assessing AUC, XAUCS
'''


'''
# from
#https://www.ibm.com/developerworks/community/blogs/jfp/entry/Fast_Computation_of_AUC_ROC_score?lang=en
# AUC-ROC = | {(i,j), i in pos, j in neg, p(i) > p(j)} | / (| pos | x | neg |)
# The equivalent version of this is, Pr [ LEFT > RIGHT ]
# now Y_true is group membership (of positive examples) , not positive level
'''
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)] #sort the predictions first
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n): # visit the examples in increasing order of predictions.
        y_i = y_true[i]
        nfalse += (1 - y_i) # negative (RIGHT) examples seen so far
        auc += y_i * nfalse # Each time we see a positive (LEFT) example we add the number of negative examples we've seen so far
    auc /= (nfalse * (n - nfalse))
    return auc
'''
cross_auc for the Ra0 > Rb1 error
function takes in scores for (a,0), (b,1)
'''
def cross_auc(R_a_0, R_b_1):
    scores = np.hstack(np.asarray([R_a_0, R_b_1]))
    y_true = np.zeros(len(R_a_0)+len(R_b_1))
    y_true[0:len(R_a_0)] = 1 # Pr[ LEFT > RIGHT]; Y = 1 is the left (A0)
    return fast_auc(y_true, scores)

'''
Use the Delong method to compute conf intervals on AUC
'''
def cross_auc_delong(R_a_0, R_b_1, alpha=0.95):
    scores = np.hstack(np.asarray([R_a_0, R_b_1]))
    y_true = np.zeros(len(R_a_0)+len(R_b_1))
    y_true[0:len(R_a_0)] = 1 # Pr[ LEFT > RIGHT]; Y = 1 is the left (A0)
    auc, auc_cov = delong_roc_variance(y_true,scores)

    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(lower_upper_q,loc=auc,scale=auc_std)
    ci[ci > 1] = 1; ci[ci < 0] = 0 # truncate interval
    return [auc, ci]

'''
Get the cross AUCs (assuming A \in \{ 0,1 \} )
'''
def get_cross_auc_delong(Rhat, Y, A, alpha=0.95):
    Rhat_a_0 = Rhat[(A==0)&(Y==0)] # a (0) is black
    Rhat_b_1 = Rhat[(A==1)&(Y==1)] # b (1) is white
    Rhat_b_0 = Rhat[(A==1)&(Y==0)] # b is white
    Rhat_a_1 = Rhat[(A==0)&(Y==1)] # a is black
    # What's the probability that a black innocent is misranked above an actually offending white?
    [xauc_a0_b1, ci_a0_b1] = cross_auc_delong(Rhat_a_0, Rhat_b_1)
    # What's the probability that a white innocent is misranked above an actually offending black?
    [xauc_b0_a1, ci_b0_a1] = cross_auc_delong(Rhat_b_0, Rhat_a_1)
    return [xauc_a0_b1, ci_a0_b1, xauc_b0_a1, ci_b0_a1]


''' Report AUCs for each level of class
'''
def get_AUCs(Rhat, Y, A):
    class_levels = np.unique(A); AUCs = np.zeros(len(class_levels))
    for ind,a in enumerate(class_levels):
        fpr, tpr, thresholds = metrics.roc_curve(Y[A==a], Rhat[A==a], pos_label=1)
        AUCs[ind] = metrics.auc(fpr,tpr)
    return AUCs
''' Report AUCs + finite conf interval for each level of class
'''
def get_AUCs_delong(Rhat,Y,A, alpha = 0.95):
    class_levels = np.unique(A); AUCs = np.zeros(len(class_levels)); CIs = [None]*len(class_levels)
    for ind,a in enumerate(class_levels):
        auc, auc_cov = delong_roc_variance(Y[A==a],Rhat[A==a])
        auc_std = np.sqrt(auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
        ci = stats.norm.ppf(lower_upper_q,loc=auc,scale=auc_std)
        ci[ci > 1] = 1; ci[ci < 0] = 0 # truncate interval
        AUCs[ind] = auc
        CIs[ind] = ci
    return [AUCs, CIs]
'''
Get the cross AUCs (assuming A \in \{ 0,1 \} )
'''
def get_cross_aucs(Rhat, Y, A, quiet=True, stump="def", save=False):
    Rhat_a_0 = Rhat[(A==0)&(Y==0)] # a (0) is black
    Rhat_b_1 = Rhat[(A==1)&(Y==1)] # b (1) is white
    Rhat_b_0 = Rhat[(A==1)&(Y==0)] # b is white
    Rhat_a_1 = Rhat[(A==0)&(Y==1)] # a is black
    if not quiet:
        plt.figure(figsize=(7,3))#  plt.figure(figsize=(3,3))
# Densities kde
        ax1 = plt.subplot(121)
        sns.set_style("white")
        sns.kdeplot(Rhat_a_0, shade=True, color = 'r', label='A=a, Y=0', clip = (0,1))
        sns.kdeplot(Rhat_b_1, shade=True, color = 'b', label='A=b, Y=1', clip = (0,1))
        plt.xlim((0,1))
        plt.title(r'KDEs of $R_A^Y$ for XAUC')
# Normed histograms
#         plt.hist(Rhat_b_1, alpha=0.5, color='blue', label='A=b, Y=1', density=True)
#         plt.hist(Rhat_a_0, alpha=0.5, color='red', label='A=a, Y=0', density=True)
        plt.legend()
#         plt.figure(figsize=(3,3))
        plt.subplot(122, sharey = ax1)
        sns.kdeplot(Rhat_b_0, shade=True, color = 'r', label='A=b, Y=0', clip = (0,1))
        sns.kdeplot(Rhat_a_1, shade=True, color = 'b', label='A=a, Y=1', clip = (0,1))
        plt.xlim((0,1))
        plt.title(r'KDEs of $R_A^Y$ for XAUC')
#         plt.hist(Rhat_b_0, alpha=0.5, color='blue', label='A=b,Y=0', density=True)
#         plt.hist(Rhat_a_1, alpha=0.5, color='red', label='A=a, Y=1', density=True)
        plt.legend()
        if save:
            plt.savefig('figs/'+stump+'KDEs.pdf')
    # What's the probability that a black innocent is misranked above an actually offending white?
    Rhata0_cross_Rhatb1 = cross_auc(Rhat_a_0, Rhat_b_1)
    # What's the probability that a white innocent is misranked above an actually offending black?
    Rhatb0_cross_Rhata1 = cross_auc(Rhat_b_0, Rhat_a_1)
    return [Rhata0_cross_Rhatb1,Rhatb0_cross_Rhata1]

''' # Get the calibration curves
'''
def get_calib_curves(Rhat, Y, A, A_labels, stump = 'def', save = False):
    plt.figure(figsize=(3,3))
    clf_scores = np.zeros(len(np.unique(A)))
    for ind,a in enumerate(np.unique(A)):
        clf_scores[ind] = brier_score_loss(Y[A==a], Rhat[A==a], pos_label=Y.max())
        fraction_of_positives, mean_predicted_value = calibration_curve(Y[A==a], Rhat[A==a] , n_bins = 10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                     label="A=%s (%1.3f)" % (A_labels[a], clf_scores[ind]))
    plt.legend()
    plt.title('Calibration curves')
    if save:
        plt.savefig('figs/'+stump+'-calibration-curves.pdf')
    return [clf_scores, mean_predicted_value, fraction_of_positives]


def get_roc(n_thresh, Rhat, Y):
    thresholds = np.linspace(1,0,n_thresh)
    ROC = np.zeros((n_thresh,2))
    for i in range(n_thresh):
        t = thresholds[i]
        # Classifier / label agree and disagreements for current threshold.
        TP_t = np.logical_and( Rhat > t, Y==1 ).sum()
        TN_t = np.logical_and( Rhat <=t, Y==0 ).sum()
        FP_t = np.logical_and( Rhat > t, Y==0 ).sum()
        FN_t = np.logical_and( Rhat <=t, Y==1 ).sum()
        # Compute false positive rate for current threshold.
        FPR_t = FP_t / float(FP_t + TN_t)
        ROC[i,0] = FPR_t
        # Compute true  positive rate for current threshold.
        TPR_t = TP_t / float(TP_t + FN_t)
        ROC[i,1] = TPR_t
    return ROC

''' Cross ROC defined for Ra0 > R1b
(permute identity of a,b, to compute the other way)
Returns a XROC with FPR on Y axis, TPR on X axis
Assume Rhat_a, Rhat_b are already separate subsets of A=a, A=b
'''
def get_cross_roc(n_thresh, Rhat_a, Rhat_b, Y_a, Y_b, A):
    thresholds = np.linspace(1,0,n_thresh)
    XROC = np.zeros((n_thresh,2))
    for i in range(n_thresh):
        t = thresholds[i]
        # Classifier / label agree and disagreements for current threshold.
        TP_t_b = np.logical_and( Rhat_b > t, Y_b==1 ).sum()
        TN_t_a = np.logical_and( Rhat_a <=t, Y_a==0 ).sum()
        FP_t_a = np.logical_and( Rhat_a > t, Y_a==0 ).sum()
        FN_t_b = np.logical_and( Rhat_b <=t, Y_b==1 ).sum()
        # Compute false positive rate for current threshold.
        FPR_t_a = FP_t_a*1.0 / (FP_t_a + TN_t_a)

        XROC[i,1] = FPR_t_a
        # Compute true  positive rate for current threshold.
        TPR_t_b = TP_t_b*1.0 / (TP_t_b + FN_t_b)
        XROC[i,0] = TPR_t_b
    return XROC



''' For now assumes that A \in \{0,1\}
'''
def get_rocs_xrocs(Rhat, Y, A, classes, n_thresh):
    ROCs_A = [None] * len(np.unique(A))
    ROCs = [ get_roc(n_thresh, Rhat[A==a], Y[A==a]) for a in np.unique(A) ]
    XROC = get_cross_roc(n_thresh, Rhat[A==0], Rhat[A==1], Y[A==0], Y[A==1], A)
    XROC_backwards = get_cross_roc(n_thresh, Rhat[A==1], Rhat[A==0],Y[A==1], Y[A==0], A)
    return [ROCs, XROC, XROC_backwards]

def plot_ROCS(ROCs, XROC, XROC_backwards, classes, A, stump = 'def', save = False):
    plt.figure(figsize=(6.5,3))
    plt.subplot(121)#
#     plt.figure(figsize=(3,3))
    [ plt.plot(ROCs[a][:,0], ROCs[a][:,1], label = classes[a]) for a in range(len(np.unique(A))) ]
    plt.legend()
    if save:
        plt.savefig('figs/'+stump+'ROC.pdf')
    plt.title('ROC curve')
    plt.subplot(122) #
#     plt.figure(figsize=(3,3))
    plt.plot(XROC[:,0], XROC[:,1], label = r'$R_a^0 > R_b^1$', color = 'blue')
    plt.plot(XROC_backwards[:,0], XROC_backwards[:,1], label = r'$R_b^0 > R_a^1$', color = 'red')
    plt.xlabel('TPR')
    plt.ylabel('FPR')
    plt.title(r'XROC curve')
    plt.legend()
    if save:
        plt.savefig('figs/'+stump+'XROC.pdf')


'''return LR model
'''
def get_lr(X,Y):
    clf = LogisticRegression(); clf.fit(X,Y)
    Rhat = clf.predict_proba(X)[:,1]
    return [clf, Rhat]

'''
!!! Main helper function
Print diagnostics for given score Rhat,
Get AUCs, XAUCs; ROC curves
'''
def get_diagnostics(Rhat, X, A, Y,labels, n_thresh, save=False,stump="default"):
    [briers, mean_predicted_value, fraction_of_positives] = get_calib_curves(Rhat, Y, A, labels, stump="stump", save=save)
    [AUCs, AUCs_CIs] = get_AUCs_delong(Rhat, Y, A)
    # print "AUCs"
    # print [ (AUCs[i], AUCs_CIs[i], labels[i]) for i in range(len(np.unique(A))) ]
    [ROCs, XROC, XROC_backwards] = get_rocs_xrocs(Rhat, Y, A, labels, n_thresh)
    plot_ROCS(ROCs, XROC, XROC_backwards, labels, A, stump = stump, save=save)
    [Rhata0_cross_Rhatb1,Rhatb0_cross_Rhata1] = get_cross_aucs(Rhat, Y,A, quiet=False, stump = stump, save=save)
    # print 'XAUCs', [Rhata0_cross_Rhatb1,Rhatb0_cross_Rhata1]

    [xauc_a0_b1, ci_a0_b1, xauc_b0_a1, ci_b0_a1] = get_cross_auc_delong(Rhat, Y, A)
    print('xauc fwds from delong', xauc_a0_b1, ci_a0_b1)
    print('xauc bwds from delong', xauc_b0_a1, ci_b0_a1)
    XAUCs=[Rhata0_cross_Rhatb1,Rhatb0_cross_Rhata1]; XCIs = [ci_a0_b1, ci_b0_a1]
    return [AUCs, AUCs_CIs, briers, ROCs, XROC, XROC_backwards, XAUCs, XCIs]


def get_calibrated_isotonic(clf, X_train,X_test, y_train):
    clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic') #clf is base estimator
    clf_isotonic.fit(X_train, y_train)
    prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]
    return [ clf_isotonic, prob_pos_isotonic ]

def get_calibrated_sigmoid(clf, X_train,X_test, y_train):
    clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method='sigmoid') #clf is base estimator
    clf_sigmoid.fit(X_train, y_train)
    prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]
    return [ clf_sigmoid, prob_pos_sigmoid ]

'''
'''


# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

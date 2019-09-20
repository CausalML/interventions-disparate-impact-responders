from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
###  for use on server
# import matplotlib as mpl
# mpl.use('Agg')
###
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import scipy.stats
from scipy import stats
import pickle
from sklearn.metrics import roc_curve, auc
import copy
import gurobipy as gp

from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from datetime import datetime
from six.moves import range




'''
This paper considers Y=1 as a benefit
'''



'''
tpr_estimation
'''
def tpr_estimation(Z, f_1pred, f_0pred, A):
    # Report in order of unique A values
    n_gps = len(np.unique(A)); tprs = np.zeros(n_gps)
    for ind,a in enumerate(np.unique(A)): # iterate over unique values of A
        num = np.mean(f_1pred[(A==a)&(Z==1)] - f_0pred[(A==a)&(Z==1)])*np.sum([(Z==1)&(A==a)])*1.0/np.sum(A==a)
        denom = np.mean(f_1pred[A==a] - f_0pred[A==a])
        tprs[ind] = num/denom
    return tprs

'''
compute ``roc curve'' based on the given score cf_pred
threshold at percentiles of cf_pred
Treat for positive response (benefit)
'''
def get_tprs_based_on_score_cutoffs(cf_pred, n_perc, f_1pred, f_0pred, A):
    cate_pctiles = np.percentile(cf_pred, np.linspace(0,100, n_perc))
    tprs__ = [ tpr_estimation((cf_pred > cfpctile).values.astype(int).flatten(), f_1pred, f_0pred, A) for cfpctile in cate_pctiles ]
    tprs = np.asarray(tprs__).reshape([len(cate_pctiles), 2])
    return [tprs, cate_pctiles]

'''
compute ``roc curve'' based on the given score cf_pred
threshold at percentiles of cf_pred
Treat for positive response (benefit)
'''
def get_tnrs_based_on_score_cutoffs(cf_pred, n_perc, f_1pred, f_0pred, A):
    cate_pctiles = np.percentile(cf_pred, np.linspace(0,100, n_perc))
    tnrs__ = [ tnr_estimation((cf_pred > cfpctile).values.astype(int).flatten(), f_1pred, f_0pred, A) for cfpctile in cate_pctiles ]
    tnrs = np.asarray(tnrs__).reshape([len(cate_pctiles), 2])
    return [tnrs, cate_pctiles]

'''
tpr_estimation
takes in a difference of treatment probabilities
'''
def tpr_estimation_delta(Z, fdelta, A):
    # Report in order of unique A values
    n_gps = len(np.unique(A)); tprs = np.zeros(n_gps)
    for ind,a in enumerate(np.unique(A)): # iterate over unique values of A
        num = np.mean(fdelta[(A==a)&(Z==1)] )*np.sum([(Z==1)&(A==a)])*1.0/np.sum(A==a)
        denom = np.mean(fdelta[A==a])
        if sum(Z==1) == 0:
            tprs[ind] = 0
        else:
            tprs[ind] = num/denom
    return tprs

'''
tpr_estimation max or min

'''
def tpr_estimation_delta_maxmin(Z, fdelta, A, upper, direction = 'max'):
    # Report in order of unique A values
    n_gps = len(np.unique(A)); tprs = np.zeros(n_gps)
    for ind,a in enumerate(np.unique(A)): # iterate over unique values of A
        tprs[ind] = tpr_closed_form_a(Z[A==a], fdelta[A==a], upper[A==a], direction)
    return tprs
def tnr_estimation_delta_maxmin(Z, fdelta, A, upper, direction = 'max'):
    # Report in order of unique A values
    n_gps = len(np.unique(A)); tprs = np.zeros(n_gps)
    for ind,a in enumerate(np.unique(A)): # iterate over unique values of A
        tprs[ind] = tnr_closed_form_a(Z[A==a], fdelta[A==a], upper[A==a], direction)
    return tprs

'''
tnr_estimation
takes in a difference of treatment probabilities
computes tnrs for each group
'''
def tnr_estimation_delta(Z, fdelta, A):
    # Report in order of unique A values
    n_gps = len(np.unique(A)); tnrs = np.zeros(n_gps)
    for ind,a in enumerate(np.unique(A)): # iterate over unique values of A
        num = np.mean(1 - fdelta[(A==a)&(Z==0)] )*np.sum([(Z==0)&(A==a)])*1.0/np.sum(A==a)
        denom = np.mean(1 - fdelta[A==a])
        if sum(Z==0) == 0:
            tnrs[ind] = 0
        else:
            tnrs[ind] = num/denom
    return tnrs

'''
plotting
'''
def plot_cate_pctiles_tprs(cate_pctiles, tprs, label0, label1, disp_curve=True):
    plt.figure()
    plt.plot(cate_pctiles,tprs[:,0],label=label0)
    plt.plot(cate_pctiles,tprs[:,1],label=label1)
    plt.figure()
    plt.plot(cate_pctiles,tprs[:,0] - tprs[:,1],label=label0)
    plt.ylabel('estimated TPR')
    plt.xlabel(r'$\alpha : Z(\alpha)= \tau(X)>\alpha$ (CATE threshold above which we treat)')

def wrapper_over_A(Z, fdelta, A, upper, ):
    (cf_pred > cfpctile).astype(int).flatten(), fdelta, A, upper

'''
get tnrs and tprs
'''
def get_tnr_tpr_max(cf_pred, n_perc, fdelta, A, upper, direction = 'max'):
    cate_pctiles = np.percentile(cf_pred, np.linspace(0,100, n_perc))

    if direction == 'max':
        tprs__ = [ tpr_estimation_delta_maxmin((cf_pred >= cfpctile).astype(int).flatten(), fdelta, A, upper, direction = 'max') for cfpctile in cate_pctiles ]
        tnrs__ = [ tnr_estimation_delta_maxmin((cf_pred >= cfpctile).astype(int).flatten(), fdelta, A, upper, direction = 'min') for cfpctile in cate_pctiles ]
    else:
        tprs__ = [ tpr_estimation_delta_maxmin((cf_pred >= cfpctile).astype(int).flatten(), fdelta, A, upper, direction = 'min') for cfpctile in cate_pctiles ]
        tnrs__ = [ tnr_estimation_delta_maxmin((cf_pred >= cfpctile).astype(int).flatten(), fdelta, A, upper, direction = 'max') for cfpctile in cate_pctiles ]
    tnrs = np.asarray(tnrs__).reshape([len(cate_pctiles), 2])
    tprs = np.asarray(tprs__).reshape([len(cate_pctiles), 2])
    return [tprs, tnrs, cate_pctiles]


'''
get tnrs and tprs
'''
def get_tnr_tpr(cf_pred, n_perc, fdelta, A, direction = 'max'):
    cate_pctiles = np.percentile(cf_pred, np.linspace(0,100, n_perc))
    tprs__ = [ tpr_estimation_delta((cf_pred >= cfpctile).astype(int).flatten(), fdelta, A) for cfpctile in cate_pctiles ]
    tprs = np.asarray(tprs__).reshape([len(cate_pctiles), 2])
    tnrs__ = [ tnr_estimation_delta((cf_pred >= cfpctile).astype(int).flatten(), fdelta, A) for cfpctile in cate_pctiles ]
    tnrs = np.asarray(tnrs__).reshape([len(cate_pctiles), 2])
    return [tprs, tnrs, cate_pctiles]


# '''
# get tnrs and tprs
# '''
# def get_tnr_tpr(cf_pred, n_perc, fdelta, A):
#     cate_pctiles = np.percentile(cf_pred, np.linspace(0,100, n_perc))
#     tprs__ = [ tpr_estimation_delta((cf_pred > cfpctile).astype(int).flatten(), fdelta, A) for cfpctile in cate_pctiles ]
#     tprs = np.asarray(tprs__).reshape([len(cate_pctiles), 2])
#     tnrs__ = [ tnr_estimation_delta((cf_pred > cfpctile).astype(int).flatten(), fdelta, A) for cfpctile in cate_pctiles ]
#     tnrs = np.asarray(tnrs__).reshape([len(cate_pctiles), 2])
#     return [tprs, tnrs, cate_pctiles]


def plot_ROC_overA(tnrs, tprs, classes, A, stump = 'def', save = False):
    plt.figure(figsize=(3,3))
    [ plt.plot(1 - tnrs[:,a], tprs[:,a], label = classes[a]) for a in range(len(np.unique(A))) ]
    plt.legend()
    if save:
        plt.savefig('figs/'+stump+'ROC.pdf')
    plt.title('ROC curve')

''' assume binary for now
'''
def plot_XROC(tnrs, tprs, classes, A, stump = 'def', xroc_labels=True, save = False, class_labels = None):
    plt.figure(figsize=(3,3))
    colors = [  'b', 'r' ]
    if xroc_labels:
        plt.plot(1 - tnrs[:,0], tprs[:,1], color = colors[0], label = r'$\tau_b^1 > \tau_a^0$')
        plt.plot(1 - tnrs[:,1], tprs[:,0], color = colors[1], label = r'$\tau_b^1 > \tau_a^0$' )
    else:
        plt.plot(1 - tnrs[:,0], tprs[:,1], color = colors[0], label = class_labels[0] )
        plt.plot(1 - tnrs[:,1], tprs[:,0], color = colors[1], label = class_labels[1] )

    plt.legend()
    if save:
        plt.savefig('figs/'+stump+'ROC.pdf')
    plt.title('XROC curve')

def plot_ROC_overA_minmax(tnrs_max, tnrs_min, tprs_max, tprs_min, classes, A, eta = 0, alpha = 1, stump = 'def', save = False):
    # plt.figure(figsize=(3,3))
    colors = [  'b', 'r' ]
    if eta > 0:
        for a in range(len(np.unique(A))):
            plt.plot(1 - tnrs_max[:,a], tprs_max[:,a], label = 'A='+str(a)+', '+ str(np.format_float_scientific(eta,precision=2)), color = colors[a], alpha = alpha)
            plt.plot(1 - tnrs_min[:,a], tprs_min[:,a], color = colors[a], alpha = alpha)
    else:
        [ plt.plot(1 - tnrs_max[:,a], tprs_max[:,a], label = classes[a], color = colors[a], alpha = alpha) for a in range(len(np.unique(A))) ]
        [ plt.plot(1 - tnrs_min[:,a], tprs_min[:,a], label = 'min' + classes[a], color = colors[a], alpha = alpha) for a in range(len(np.unique(A))) ]

    # plt.legend()
    if save:
        plt.savefig('figs/'+stump+'ROC-bounds.pdf')
    plt.title('robust ROC curve')


def plot_XROC_overA_minmax(tnrs_max, tnrs_min, tprs_max, tprs_min, classes, A, alpha = 1, stump = 'def', save = False):
    # plt.figure(figsize=(3,3))
    colors = [  'b', 'r' ]

    plt.plot(1 - tnrs_max[:,0], tprs_max[:,1], color = colors[0], alpha = alpha)
    plt.plot(1 - tnrs_min[:,0], tprs_min[:,1], color = colors[0], alpha = alpha)

    plt.plot(1 - tnrs_max[:,1], tprs_max[:,0], color = colors[1], alpha = alpha)
    plt.plot(1 - tnrs_min[:,1], tprs_min[:,0], color = colors[1], alpha = alpha)
    # plt.legend()
    if save:
        plt.savefig('figs/'+stump+'XROC-bounds.pdf')
    plt.title('robust ROC curve')


'''
Helper functions for estimating propensities
'''
def estimate_prop(x, T, predict_x, predict_T):
    clf_dropped = LogisticRegression()
    clf_dropped.fit(x, T)
    est_prop = clf_dropped.predict_proba(predict_x)
    est_Q = np.asarray( [est_prop[k,1] if predict_T[k] == 1 else est_prop[k,0] for k in range(len(predict_T))] )
    return [est_Q, clf_dropped]

    # get indicator vector from signed vector
def get_0_1_sgn(vec):
    n = len(vec)
    return np.asarray([1 if vec[i] == 1 else 0 for i in range(n) ]).flatten()
# get signed vector from indicator vector
def get_sgn_0_1(vec):
    n = len(vec)
    return np.asarray([1 if vec[i] == 1 else -1 for i in range(n) ]).flatten()

''' Assuming T \in 0,1
Get the prediction models of response within each treated/untreated group
'''
def fit_clfs(X_jobs, T_jobs, Y_jobs):
    clf_0 = LogisticRegression(); clf_0.fit(X_jobs[T_jobs==0,:], Y_jobs[T_jobs==0])
    prob_Y0 = clf_0.predict_proba(X_jobs[T_jobs==0,:])
    clf_1 = LogisticRegression(); clf_1.fit(X_jobs[T_jobs==1,:], Y_jobs[T_jobs==1])
    prob_Y1 = clf_1.predict_proba(X_jobs[T_jobs==1,:])
    return [clf_0, clf_1, prob_Y0, prob_Y1]



# ''' weights
# default behavior: maximize
# return primal weights
# return scaling t
# a,b bounds in space relative to score values
# '''
# def opt_flp(y,a,b,quiet=True):
#     m = gp.Model()
#     if quiet: m.setParam("OutputFlag", 0)
#     t = m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
#     w = [m.addVar(obj = -yy, lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
#     m.update()
#     m.addConstr(gp.quicksum(w)==1)
#     for i in range(len(y)):
#         m.addConstr(w[i] <= b[i] * t)
#         m.addConstr(w[i] >= a[i] * t)
#     m.optimize()
#     if (m.status == gp.GRB.OPTIMAL):
#         wghts = np.asarray([ ww.X for ww in w ])
#         return [-m.ObjVal, wghts, t.X]
#     else:
#         return [np.nan, 0, 0]


''' weights
default behavior: maximize
return primal weights
return scaling t
a,b bounds centered at 0
'''
def opt_flp_normalized(y,b,sum_fdelta_num,sum_fdelta_denom,direction = 'max',quiet=True):
    m = gp.Model()
    if quiet: m.setParam("OutputFlag", 0)
    t = m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    w = [m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
    m.update()
    m.addConstr(gp.quicksum(w) + t*sum_fdelta_denom==1)
    for i in range(len(y)):
        m.addConstr(w[i] <= b[i] * t)
    if direction == 'max':
        m.setObjective(gp.quicksum( w[i]*y[i] for i in range(len(y)) ) + t*sum_fdelta_num, gp.GRB.MAXIMIZE); m.optimize()
    elif direction == 'min':
        m.setObjective(gp.quicksum( w[i]*y[i] for i in range(len(y)) ) + t*sum_fdelta_num, gp.GRB.MINIMIZE); m.optimize()
    m.optimize()
    if (m.status == gp.GRB.OPTIMAL):
        wghts = np.asarray([ ww.X for ww in w ])
        return [m.ObjVal, wghts, t.X]
    else:
        return [np.nan, 0, 0]

''' weights
default behavior: maximize
Intended for optimizing TNR type disparities
We optimize over TNR by replacing
sum_fdelta_num by n_{Z=0} - sum fdelta Z
sum_fdelta_denom by n - sum fdelta
since we multiply the objective function variable by -1
return primal weights
return scaling t
a,b bounds centered at 0
'''
def opt_flp_normalized_TNR(y,b,sum_fdelta_num,sum_fdelta_denom,direction = 'max',quiet=True):
    m = gp.Model()
    if quiet: m.setParam("OutputFlag", 0)
    t = m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    w = [m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
    m.update()
    m.addConstr(-1*gp.quicksum(w) + t*sum_fdelta_denom==1)
    for i in range(len(y)):
        m.addConstr(w[i] <= b[i] * t)
    if direction == 'max':
        m.setObjective(gp.quicksum( -1 * w[i]*y[i] for i in range(len(y)) ) + t*sum_fdelta_num, gp.GRB.MAXIMIZE); m.optimize()
    elif direction == 'min':
        m.setObjective(gp.quicksum( -1 * w[i]*y[i] for i in range(len(y)) ) + t*sum_fdelta_num, gp.GRB.MINIMIZE); m.optimize()
    m.optimize()
    if (m.status == gp.GRB.OPTIMAL):
        wghts = np.asarray([ ww.X for ww in w ])
        return [m.ObjVal, wghts, t.X]
    else:
        return [np.nan, 0, 0]

def tpr_closed_form_a(Z, fdelta, upper, direction = 'max', ret_over_A = False, ret_weights=False):
    r_a_num = np.sum(fdelta*Z);
    r_a_denom = np.sum(fdelta);
    b = upper
    # if sum(Z==1) == 0:
    #     return 0
    if direction=='max':
        return (r_a_num + np.sum(b[(Z==1)]) ) / (r_a_denom + np.sum(b[(Z==1)]))
    else:
        return (r_a_num ) / (r_a_denom + np.sum(b[(Z==0)]) )

def tnr_closed_form_a(Z, fdelta, upper, direction = 'max', ret_over_A = False, ret_weights=False):
    r_a_num = np.sum((Z==0)) - np.sum(fdelta[(Z==0)]);
    r_a_denom = len(Z) - np.sum(fdelta)
    b = upper
    # if sum(Z==0) == 0:
    #     return
    if direction=='max':
        return (r_a_num ) / (r_a_denom - np.sum(b[(Z==1)]))
    else:
        return (r_a_num - np.sum(b[(Z==0)])) / (r_a_denom - np.sum(b[(Z==0)]))

''' tpr closed form
Max tpr by the closed form solution
'''
def tpr_closed_form(Z, A, fdelta, upper, direction = 'max', ret_over_A = False, ret_weights=False):
    a_ind = np.where(A==0)[0]; b_ind = np.where(A==1)[0]; Z_a = Z[a_ind]; Z_b = Z[b_ind]
    r_a_num = np.sum(fdelta[a_ind]*Z_a);r_b_num = np.sum(fdelta[b_ind]*Z_b) # constant terms
    r_a_denom = np.sum(fdelta[a_ind]);r_b_denom = np.sum(fdelta[b_ind])
    b = upper
    if direction=='max':
        obj_a = (r_a_num + np.sum(b[(A==0)&(Z==1)])) / (r_a_denom + np.sum(b[(A==0)&(Z==1)]))
        obj_b = (r_b_num ) / (r_b_denom + np.sum(b[(A==1)&(Z==0)]))
        return obj_a - obj_b
    else:
        obj_a = (r_a_num ) / (r_a_denom + np.sum(b[(A==0)&(Z==0)]) )
        obj_b = (r_b_num + np.sum(b[(A==1)&(Z==1)])) / (r_b_denom + np.sum(b[(A==1)&(Z==1)]))
        return obj_a - obj_b

''' tnr closed form
'''
def tnr_closed_form_fn_alpha(Z, A, fdelta, upper, direction = 'max', ret_weights=False):
    r_a_num = np.sum((Z==0)&(A==0)) - np.sum(fdelta[(Z==0)&(A==0)]);
    r_a_denom = np.sum(A==0) - np.sum(fdelta[A==0])
    r_b_num = np.sum((Z==0)&(A==1)) - np.sum(fdelta[(Z==0)&(A==1)]);
    r_b_denom = np.sum(A==1) - np.sum(fdelta[A==1])
    B = upper; B_ = np.max(upper); n_alphas = 50
    alphas = np.linspace(0,B_,n_alphas)
    plt.plot(alphas, [(r_a_num -alpha*np.sum((A==0)&(Z==0))) / (r_a_denom - alpha*np.sum((A==0)&(Z==0)) - np.sum(B[(A==0)&(Z==1)])) for alpha in alphas] )
    # plt.plot( [(r_b_num -alpha*np.sum((A==1)&(Z==0))) / (r_b_denom - alpha*np.sum((A==1)&(Z==0)) + np.sum(B[(A==1)&(Z==1)])) for alpha in alphas] )



''' tnr closed form
'''
def tnr_closed_form(Z, A, fdelta, upper, direction = 'max', ret_weights=False):
    r_a_num = np.sum((Z==0)&(A==0)) - np.sum(fdelta[(Z==0)&(A==0)]);
    r_a_denom = np.sum(A==0) - np.sum(fdelta[A==0])
    r_b_num = np.sum((Z==0)&(A==1)) - np.sum(fdelta[(Z==0)&(A==1)]);
    r_b_denom = np.sum(A==1) - np.sum(fdelta[A==1])
    B = upper
    if direction=='max':
        # if B*np.sum((A==0)&(Z==1)) + np.sum( fdelta[(Z==1)&(A==0)] ) - sum((Z==1))
        obj_a = (r_a_num ) / (r_a_denom - np.sum(B[(A==0)&(Z==1)]))
        obj_b = (r_b_num - np.sum(B[(A==1)&(Z==0)])) / (r_b_denom - np.sum(B[(A==1)&(Z==0)]))
    else:
        obj_a = (r_a_num - np.sum(B[(A==0)&(Z==0)])) / (r_a_denom - np.sum(B[(A==0)&(Z==0)]))
        obj_b = (r_b_num ) / (r_b_denom - np.sum(B[(A==1)&(Z==1)]))
    return obj_a - obj_b

'''
compute max tnr_a - tnr_b
s.t. fdelta < upper
We optimize over TNR by replacing
sum_fdelta_num by n_{Z=0} - sum fdelta Z
sum_fdelta_denom by n - sum fdelta
'''
def tnr_disparity(Z, A, fdelta, upper, direction = 'max', ret_weights=False):
    a_ind = np.where(A==0)[0]; b_ind = np.where(A==1)[0]; Z_a = Z[a_ind]; Z_b = Z[b_ind]
    # r_a_num = np.sum(fdelta[a_ind]*Z_a);r_b_num = np.sum(fdelta[b_ind]*Z_b) # constant terms
    # r_a_denom = np.sum(fdelta[a_ind]);r_b_denom = np.sum(fdelta[b_ind])
    r_a_num = np.sum((Z==0)&(A==0)) - np.sum(fdelta[(Z==0)&(A==0)]);
    r_a_denom = np.sum(A==0) - np.sum(fdelta[A==0])
    r_b_num = np.sum((Z==0)&(A==1)) - np.sum(fdelta[(Z==0)&(A==1)]);
    r_b_denom = np.sum(A==1) - np.sum(fdelta[A==1])
    if direction == 'max':
                # num = np.mean(1 - fdelta[(A==a)&(Z==0)] )*np.sum([(Z==0)&(A==a)])*1.0/np.sum(A==a)
                # denom = np.mean(1 - fdelta[A==a])
        [obj_a, wghts_a, t_a] = opt_flp_normalized_TNR( 1-Z[A==0], upper[A==0],
        np.sum((Z==0)&(A==0)) - np.sum(fdelta[(Z==0)&(A==0)]), np.sum(A==0) - np.sum(fdelta[A==0]), direction = 'max')
        [obj_b, wghts_b, t_b] = opt_flp_normalized_TNR( 1-Z[A==1], upper[A==1],
        np.sum((Z==0)&(A==1)) - np.sum(fdelta[(Z==0)&(A==1)]), np.sum(A==1) - np.sum(fdelta[A==1]), direction = 'min')
    elif direction == 'min':
        [obj_a, wghts_a, t_a] = opt_flp_normalized_TNR( 1-Z[A==0], upper[A==0],
        np.sum((Z==0)&(A==0)) - np.sum(fdelta[(Z==0)&(A==0)]), np.sum(A==0) - np.sum(fdelta[A==0]), direction = 'min')
        [obj_b, wghts_b, t_b] = opt_flp_normalized_TNR( 1-Z[A==1], upper[A==1],
        np.sum((Z==0)&(A==1)) - np.sum(fdelta[(Z==0)&(A==1)]), np.sum(A==1) - np.sum(fdelta[A==1]), direction = 'max')
    if not ret_weights:
        return obj_a - obj_b
    else:
        return [obj_a - obj_b, wghts_a, wghts_b, t_a, t_b ]

'''
compute max tpr_a - tpr_B
s.t. fdelta < upper
'''
def tpr_disparity(Z, A, fdelta, upper, direction = 'max', ret_weights=False):
    a_ind = np.where(A==0)[0]; b_ind = np.where(A==1)[0]; Z_a = Z[a_ind]; Z_b = Z[b_ind]
    r_a_num = np.sum(fdelta[a_ind]*Z_a);r_b_num = np.sum(fdelta[b_ind]*Z_b) # constant terms
    r_a_denom = np.sum(fdelta[a_ind]);r_b_denom = np.sum(fdelta[b_ind])
    if direction == 'max':
        [obj_a, wghts_a, t_a] = opt_flp_normalized( Z[A==0], upper[A==0], r_a_num, r_a_denom, direction = 'max')
        [obj_b, wghts_b, t_b] = opt_flp_normalized( Z[A==1], upper[A==1], r_b_num, r_b_denom, direction = 'min')
    elif direction == 'min':
        [obj_a, wghts_a, t_a] = opt_flp_normalized( Z[A==0], upper[A==0], r_a_num, r_a_denom, direction = 'min')
        [obj_b, wghts_b, t_b] = opt_flp_normalized( Z[A==1], upper[A==1], r_b_num, r_b_denom, direction = 'max')
    if not ret_weights:
        return obj_a - obj_b
    else:
        return [obj_a - obj_b, wghts_a, wghts_b, t_a, t_b ]

'''
compute disparity curve based on the given score cf_pred
threshold at percentiles of cf_pred
Treat for positive response (benefit)
inputs:
tau_pred: prediction of CATE (score)
n_perc: number of percentiles to threshold on
fdelta: within sample predictions
A: race
upper: upper bound on W
direction: compte max or min roc curves
'''
def get_tpr_disps_based_on_score_cutoffs(tau_pred, n_perc, fdelta, A, upper, direction='max', parallel = True):
    cate_pctiles = np.percentile(tau_pred, np.linspace(0,100, n_perc))
    if parallel:
        res_ = Parallel(n_jobs=12, verbose = 10)(delayed(tpr_disparity)((tau_pred > cfpctile).astype(int), A, fdelta, upper, direction) for cfpctile in cate_pctiles )
    else:
        res_ = [ tpr_disparity((tau_pred > cfpctile).astype(int), A, fdelta, upper, direction) for cfpctile in cate_pctiles ]
    return [np.asarray(res_), cate_pctiles]




'''
get disparity curve over different values of eta
Specify the disparity function with "disparity"
Default to closed form solutions
'''
def get_disp_curve_over_etas(tau_hat, n_perc, etas, A, fdelta, plotting=True,
disparity = tpr_closed_form, FILE_LOC='disp_curve_etas_tpr.pdf', parallel = "closed_form"):
    cate_pctiles = np.percentile(tau_hat, np.linspace(0,100, n_perc))
    if parallel=='parallel':
        res_upper = Parallel(n_jobs=12, verbose = 10)(delayed(disparity)((tau_hat > cfpctile).astype(int), A, fdelta, eta__*np.ones(len(A)), direction='max') for cfpctile in cate_pctiles for eta__ in etas )
        res_lower = Parallel(n_jobs=12, verbose = 10)(delayed(disparity)((tau_hat > cfpctile).astype(int), A, fdelta, eta__*np.ones(len(A)), direction='min') for cfpctile in cate_pctiles for eta__ in etas )
    else:
        res_upper = [ disparity((tau_hat > cfpctile).astype(int), A, fdelta, eta__*np.ones(len(A)), direction='max') for cfpctile in cate_pctiles for eta__ in etas ]
        res_lower = [ disparity((tau_hat > cfpctile).astype(int), A, fdelta, eta__*np.ones(len(A)), direction='min') for cfpctile in cate_pctiles for eta__ in etas ]
    # reshape output from Parallel
    res_lower_ = np.asarray(res_lower).reshape([len(cate_pctiles), len(etas)]);
    res_upper_ = np.asarray(res_upper).reshape([len(cate_pctiles), len(etas)])
    alphas = np.linspace(1,0.1, len(etas))
    if plotting:
        plt.figure(figsize=(4.5,3));plt.axhline(0, color='black')
        [plt.plot( cate_pctiles, res_lower_[:,i], label=etas[i], color = 'b', alpha = alphas[i] ) for i in range(len(etas))]
        [plt.plot( cate_pctiles, res_upper_[:,i], label=etas[i], color = 'r', alpha = alphas[i] ) for i in range(len(etas))]
        plt.ylabel(r'$\Delta^{TPR}_{(a,b)}$')
        plt.xlabel(r'$\tau$ threshold')
        plt.savefig(FILE_LOC)
    return [res_lower_, res_upper_]



'''
# tpr disparities
Plot tpr disparities
'''
def get_tpr_disps(tau_hat, n_perc, fdelta, A, upper_fdelta, plotting=True):
    [max_tprs_disps, cate_pctiles] = get_tpr_disps_based_on_score_cutoffs(tau_hat, n_perc, fdelta, A, upper_fdelta)
    [min_tprs_disps, cate_pctiles] = get_tpr_disps_based_on_score_cutoffs(tau_hat, n_perc, fdelta, A, upper_fdelta, direction='min')
    if plotting:
        plt.plot(cate_pctiles, max_tprs_disps)
        plt.plot(cate_pctiles, min_tprs_disps)
        plt.axhline(0, color='black')
    return [max_tprs_disps, min_tprs_disps, cate_pctiles]



'''
Get increasing curves of TPR disparities
over different values of eta
'''
def get_tpr_disps_over_eta(etas, tau_hat, n_perc, fdelta, A, upper_fdelta, plotting=True, vbs = 5, disparity=tpr_disparity, save = 'out/'):
    cate_pctiles = np.percentile(tau_hat, np.linspace(0,100, n_perc))
    res_upper = Parallel(n_jobs=12, verbose = vbs)(delayed(disparity)((tau_hat > cfpctile).astype(int), A, fdelta, eta__*np.ones(len(A)), direction='max') for cfpctile in cate_pctiles for eta__ in etas )
    res_lower = Parallel(n_jobs=12, verbose = vbs)(delayed(disparity)((tau_hat > cfpctile).astype(int), A, fdelta, eta__*np.ones(len(A)), direction='min') for cfpctile in cate_pctiles for eta__ in etas )
    res_lower_ = np.asarray(res_lower).reshape([len(cate_pctiles), len(etas)])
    res_upper_ = np.asarray(res_upper).reshape([len(cate_pctiles), len(etas)])
    alphas = np.linspace(1,0.1, len(etas))
    if plotting:
        plt.figure(figsize=(4,4));plt.axhline(0, color='black')
        [plt.plot( cate_pctiles, res_lower_[:,i], label=etas[i], color = 'b', alpha = alphas[i] ) for i in range(len(etas))]
        [plt.plot( cate_pctiles, res_upper_[:,i], label=etas[i], color = 'r', alpha = alphas[i] ) for i in range(len(etas))]
        plt.ylabel(r'$\Delta^{TPR}_{(a,b)}$')
        plt.xlabel(r'$\tau$ threshold')
        plt.legend(bbox_to_anchor=(1.1,1))
        plt.savefig(save+disparity+'curve.pdf')
    return [ res_lower_, res_upper_]

def opt_flp_tpr_disp_fixed_t_val(t, LAMBDA, fdelta, A, Z, b_,
            quiet=True, smoothness = False, dists_mat = False, LIPSCH_CONST=False, direction='max'):
    return opt_flp_tpr_disp_fixed_t(t, LAMBDA, fdelta, A, Z, b_,
                quiet=True, smoothness = smoothness, dists_mat = dists_mat, LIPSCH_CONST=LIPSCH_CONST, direction=direction)[0]

def ols_formula(df, dependent_var, top_level = True, excluded_cols = [None]):
    '''
    Generates the R style formula for statsmodels (patsy) given
    the dataframe, dependent variable and optional excluded columns
    as strings
    '''
    df_columns = list(df.columns.values)
    df_columns.remove(dependent_var)
    for col in excluded_cols:
        df_columns.remove(col)
    if top_level:
        return dependent_var + ' ~ ' + ' + '.join(df_columns)
    else:
        return ' + '.join(df_columns)


'''
helper for single call for greater parallelization
return list of result objects
'''
def opt_smoothness_within_partitions( Z_, X, fdelta, A, upper, LIPSCH_CONST, plotting = False):
    fdelta_A0 = fdelta[A==0]; w_upper_A0 = upper[A==0]; Z_A0 = Z_[A==0]; [dists_mat_A0, X_A0] = dists_matrix(X[A==0,:])
    # upper bound of Lipschitz constant: less than 1 /
    res_upper_A0 = opt_flp_smoothness_n2(Z_A0, w_upper_A0, sum(fdelta[(Z_==1)&(A==0)]), sum(fdelta[(A==0)]),dists_mat_A0, LIPSCH_CONST,quiet=True, direction ='max')
    res_lower_A0 = opt_flp_smoothness_n2(Z_A0, w_upper_A0, sum(fdelta[(Z_==1)&(A==0)]), np.sum(fdelta[(A==0)]),dists_mat_A0, LIPSCH_CONST,quiet=True, direction ='min')
    # Get max smoothness disparity and use the achieving weights to calibrate $\Lambda$
    fdelta_A1 = fdelta[A==1];w_upper_A1 = upper[A==1];Z_A1 = Z_[A==1];[dists_mat_A1, X_A1] = dists_matrix(X[A==1,:])
    res_upper_A1 = opt_flp_smoothness_n2(Z_A1, w_upper_A1, np.sum(fdelta[(Z_==1)&(A==1)]), np.sum(fdelta[(A==1)]), dists_mat_A1, LIPSCH_CONST,quiet=True, direction ='max')
    res_lower_A1 = opt_flp_smoothness_n2(Z_A1, w_upper_A1, np.sum(fdelta[(Z_==1)&(A==1)]), np.sum(fdelta[(A==1)]), dists_mat_A1, LIPSCH_CONST,quiet=True, direction ='min')
    res_lower_uppers = [ res_lower_A0, res_upper_A0, res_lower_A1, res_upper_A1]
    return res_lower_uppers

'''
get disparity curve over different values of eta
for smoothness within each partition

'''
def get_disp_curve_over_etas_smoothness(tau_hat, n_perc, X, fdelta, A, etas, LIPSCH_CONST, plotting=True, FILE_LOC='disp_curve_etas_tpr_smoothness.pdf'):
    cate_pctiles = np.percentile(tau_hat, np.linspace(0,100, n_perc))
    res_ = Parallel(n_jobs=12, verbose = 10)(delayed(opt_smoothness_within_partitions)((tau_hat > cfpctile).astype(int), X, fdelta, A, eta__*np.ones(len(A)), LIPSCH_CONST ) for cfpctile in cate_pctiles for eta__ in etas )
    pickle.dump(res_, open('disp_curve_over_etas_smoothness'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.pkl', 'wb'))
    objs__ = np.zeros([len(res_),4])
    eff_lambdas_max = np.zeros(len(res_)); eff_lambdas_min = np.zeros(len(res_));
    for i in range(len(res_)):
        objs__[i, :] = [res_[i][j][0] for j in range(len(res_[i]))]
        eff_lambdas_max[i] = np.sum(res_[i][1][1])/res_[i][1][2] + np.sum(res_[i][2][1])/res_[i][2][2]
        eff_lambdas_min[i] = np.sum(res_[i][0][1])/res_[i][0][2] + np.sum(res_[i][3][1])/res_[i][3][2]
    # reshape output from Parallel
    res_lower_A0 = objs__[:,0].reshape([len(cate_pctiles), len(etas)]);
    res_upper_A0 = objs__[:,1].reshape([len(cate_pctiles), len(etas)])
    res_lower_A1 = objs__[:,2].reshape([len(cate_pctiles), len(etas)]);
    res_upper_A1 = objs__[:,3].reshape([len(cate_pctiles), len(etas)]);
    eff_lambdas_max = eff_lambdas_max.reshape([len(cate_pctiles), len(etas)])
    eff_lambdas_min = eff_lambdas_min.reshape([len(cate_pctiles), len(etas)])
    alphas = np.linspace(1,0.1, len(etas))
    if plotting:
        fig = plt.figure(figsize=(4,4));plt.axhline(0, color='black')
        [plt.plot( cate_pctiles, res_upper_A0[:,i] - res_lower_A1[:,i], label=etas[i], color = 'r', alpha = alphas[i] ) for i in range(len(etas))]
        [plt.plot( cate_pctiles, res_lower_A0[:,i] - res_upper_A1[:,i], label=etas[i], color = 'b', alpha = alphas[i] ) for i in range(len(etas))]
        plt.ylabel(r'$\Delta^{TPR}_{(a,b)}$')
        plt.xlabel(r'$\tau$ threshold')
        plt.savefig(FILE_LOC)
    return [ res_lower_A0, res_upper_A0, res_lower_A1, res_upper_A1, res_, fig, eff_lambdas_max, eff_lambdas_min]

'''
given a list of results
get "effective lambdas" for each
assume original program returns nonhomogenized weights
'''
def get_eff_lambda(res, w_upper):
    eff_lambdas = np.zeros(len(res))
    for ind,x in enumerate(res):
        if (~np.isnan(x[0])):
            wghts = x[1]; t = x[2]
#             print wghts /t  - w_lower
            eff_lambdas[ind] = np.sum(wghts /t  - w_lower)
    return eff_lambdas

'''
Compute effective Lambdas achieved by results for different upper, lower bounds

'''
def get_eff_lambdas_results_LCs(res_upper_A0, res_lower_A0, res_upper_A1, res_lower_A1, w_lower_A0, w_upper_A0, w_lower_A1, w_upper_A1):
    eff_lambda_A1 = get_eff_lambda(res_upper_A1, w_upper_A1)
    eff_lambda_A1_lower = get_eff_lambda(res_lower_A1, w_lower_A1)
    # print eff_lambda_A1
    # print 'effective lambda A1, lower', eff_lambda_A1_lower
    eff_lambda_A0 = get_eff_lambda(res_upper_A0, w_upper_A0)
    eff_lambda_A0_lower = get_eff_lambda(res_lower_A0, w_lower_A0)
    # print eff_lambda_A0
    # print 'effective lambda A1, lower', eff_lambda_A0_lower
    return [ eff_lambda_A0, eff_lambda_A0_lower, eff_lambda_A1, eff_lambda_A1_lower]


'''
'''
def compare_def_smooth_TPR_disp(tau_hat, n_perc, X_llnde, fdelta, A, etas, LIPSCH_CONST):

    cate_pctiles = np.percentile(tau_hat, np.linspace(0,100, n_perc))

    [ res_lower_A0, res_upper_A0, res_lower_A1, res_upper_A1, res_, fig] = get_disp_curve_over_etas_smoothness(tau_hat, n_perc, X_llnde, fdelta, A, etas, LIPSCH_CONST)
    [res_lower_, res_upper_] = get_disp_curve_over_etas(tau_hat, n_perc, etas, A, fdelta, plotting=True)
    for j in range(len(etas)):
        plt.figure()
        plt.plot(cate_pctiles, res_upper_[:,j])
        plt.plot(cate_pctiles, res_lower_[:,j])
        plt.plot(cate_pctiles, res_upper_[:,j])
        plt.plot(cate_pctiles, res_lower_[:,j])
        plt.plot(cate_pctiles,res_upper_A0[:,j] - res_lower_A1[:,j])
        plt.plot(cate_pctiles,res_lower_A0[:,j] - res_upper_A1[:,j])


'''
get disparity curve over different values of eta
for smoothness within each partition
Calls as a subroutine the disparity function with signature
'opt_flp_tpr_disp_over_t(ts, LAMBDA, fdelta, A, Z, b_,
            quiet=True, smoothness = False, dists_mat = False, LIPSCH_CONST=False,
            direction='max')'
'''
def get_disp_curve_over_etas_Budgeted_Smoothness(tau_hat, n_perc, ts,
LMBDA_PERC, fdelta, A, Z, b_, eff_lambdas, etas, quiet=True, smoothness = True,
dists_mat = None, LIPSCH_CONST=None, direction='max',
FILE_LOC='disp_curve_etas_tpr_budgeted_smoothness.pdf', disp_str = 'TPR',
N_JOBS = 12, vbs = 10, plotting= True):
    cate_pctiles = np.percentile(tau_hat, np.linspace(0,100, n_perc))
    res_ = Parallel(n_jobs=N_JOBS, verbose = vbs)(delayed(opt_flp_tpr_disp_over_t_wrapper)(ts,
    LMBDA_PERC*eff_lambdas[pct_ind, eta_ind], fdelta, A,
    (tau_hat > cate_pctiles[pct_ind]).astype(int), etas[eta_ind]*np.ones(len(A)),
    smoothness=True, dists_mat = dists_mat, LIPSCH_CONST=LIPSCH_CONST,
    direction = direction) for pct_ind in range(len(cate_pctiles)) for eta_ind in range(len(etas)) )
    pickle.dump(res_, open('disp_curve_over_etas_budgeted_smoothness-'+direction+'-'+disp_str +datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.pkl', 'wb'))
    objs__ = np.asarray([x[0] for x in res_]).reshape([len(cate_pctiles), len(etas)]);
    return [ objs__, res_]

def plot_disps_over_eta(cate_pctiles, etas, res_upper, res_lower, disp_str = 'tpr', FILE_LOC='out/'):
    alphas = np.linspace(1,0.1, len(etas))
    fig = plt.figure(figsize=(4,4));plt.axhline(0, color='black')
    [plt.plot( cate_pctiles, res_upper, label=etas[i], color = 'r', alpha = alphas[i] ) for i in range(len(etas))]
    [plt.plot( cate_pctiles, res_lower, label=etas[i], color = 'b', alpha = alphas[i] ) for i in range(len(etas))]
    plt.ylabel(r'$\Delta^{'+disp_str+'}_{(a,b)}$')
    plt.xlabel(r'$\tau$ threshold')
    plt.savefig(FILE_LOC+'disps_over_eta.pdf')

'''
convention to keep headers on covariate data because we need to index on column
'''
def read_data( XTY_locs ):
    X = pd.read_csv(XTY_locs[0]).values
    T = pd.read_csv(XTY_locs[1], header=None).values.flatten()
    Y = pd.read_csv(XTY_locs[2], header=None).values.flatten()
    A = pd.read_csv(XTY_locs[3], header=None).values.flatten()
    tau_hat = pd.read_csv(XTY_locs[4]).values.flatten()
    data = [X,T,Y, tau_hat];
    # print [ len(d) for d in data]
    [clf_0, clf_1, prob_Y0, prob_Y1] = fit_clfs(X, T, Y)
    f_1pred = clf_1.predict_proba(X)[:,1]
    f_0pred = clf_0.predict_proba(X)[:,1]
    fdelta = np.clip(f_1pred - f_0pred, 0, 1)
    return [X,T,Y,A,tau_hat,f_1pred,f_0pred, fdelta]

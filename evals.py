import numpy as np
import sys
sys.path.append('./POEM-norm/')
# TOFIX: bad hack for absolute imports
sys.path.append('/Users/az/Box Sync/unconfoundedness/POEM-norm')
import DatasetReader, Skylines

def get_oos_est_anti(pi_test):
    return np.mean(y_test[pi_test==t_test] / true_Q_test[pi_test==t_test]) -np.mean(y_test[pi_test!=t_test] / true_Q_test[pi_test!=t_test] )

def get_oos_est_prob(pi_1):
    t_=pi_1 > 0.5
    return np.mean(y_test[t_==t_test] / true_Q_test[t_==t_test]) -np.mean(y_test[t_!=t_test] / true_Q_test[t_!=t_test] )

# rescale each numerical column separately?
def transform_categorical_frame(cat_df, cov_types):
    cols = [ c for c in cat_df.columns ]
    scales = []
    for ind,typ in enumerate(cov_types):
        if typ =='num':
            mn = cat_df[cols[ind]].mean(); std=cat_df[cols[ind]].std()
            scales += [(mn, std)]
            cat_df[cols[ind]] = (cat_df[cols[ind]] - mn)/std
    return [cat_df, scales]

def transform_test(X_cat_test, scales, cov_types):
    cols = [ c for c in cat_df.columns ]
    for ind,typ in enumerate(cov_types):
        if typ =='num':
            cat_df[cols[ind]] = (cat_df[cols[ind]] - scales[i][0])/scales[i][1]
    return [cat_df, scales]

def get_POEM_rec(x,t,y,nominal_Q,x_test):
    mydata = DatasetReader.BanditDataset(None,False)
    mydata.trainFeatures        = np.hstack((x.copy(),np.ones((len(x),1))))
    mydata.sampledLabels        = np.zeros((len(t),max(t)+1))
    mydata.sampledLabels[range(len(t)),t] = 1.
    mydata.trainLabels          = np.empty(mydata.sampledLabels.shape)
    mydata.sampledLoss          = y.copy()
    mydata.sampledLoss         -= mydata.sampledLoss.min()
    mydata.sampledLoss         /= mydata.sampledLoss.max()
    # computed on training set
    mydata.sampledLogPropensity = np.log(nominal_Q)
    #ones_like vs ones_line?
    mydata.testFeatures              = np.hstack((np.ones_like(x_test),np.ones((len(x_test),1))))
    mydata.testLabels                = np.array([])
    mydata.createTrainValidateSplit()
    pool = None
    coef = None
    maj = Skylines.PRMWrapper(mydata, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = 0, maxV = -1,
                                minClip = 0, maxClip = 0, estimator_type = 'Vanilla', verbose = True,
                                parallel = pool, smartStart = coef)
    maj.calibrateHyperParams()
    maj.validate()
    Xtest1 = np.hstack((x_test,np.ones((len(x_test),1))))
    rec = Xtest1.dot(maj.labeler.coef_).argmax(1)

    return [maj,rec]


def get_SNPOEM_rec(x,t,y,nominal_Q,x_test):
    mydata = DatasetReader.BanditDataset(None,False)
    mydata.trainFeatures        = np.hstack((x.copy(),np.ones((len(x),1))))
    mydata.sampledLabels        = np.zeros((len(t),max(t)+1))
    mydata.sampledLabels[range(len(t)),t] = 1.
    mydata.trainLabels          = np.empty(mydata.sampledLabels.shape)
    mydata.sampledLoss          = y.copy()
    mydata.sampledLoss         -= mydata.sampledLoss.min()
    mydata.sampledLoss         /= mydata.sampledLoss.max()
    # computed on training set
    mydata.sampledLogPropensity = np.log(nominal_Q)
    #ones_like vs ones_line?
    mydata.testFeatures              = np.hstack((np.ones_like(x_test),np.ones((len(x_test),1))))
    mydata.testLabels                = np.array([])
    mydata.createTrainValidateSplit()
    pool = None
    coef = None
    maj = Skylines.PRMWrapper(mydata, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = 0, maxV = -1,
                                minClip = 0, maxClip = 0, estimator_type = 'Self-Normalized', verbose = True,
                                parallel = pool, smartStart = coef)
    maj.calibrateHyperParams()
    maj.validate()
    Xtest1 = np.hstack((x_test,np.ones((len(x_test),1))))
    rec = Xtest1.dot(maj.labeler.coef_).argmax(1)

    return [maj,rec]

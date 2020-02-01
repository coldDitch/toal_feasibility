"""
       Python functions for RBF stan models

       bayesname: name of the model
       stan_folder: folder for compiled stan models
       n: number real datapoints
       x: covariates
       a: actions
       y: responses
       cn: number of censored datapoints
       cx: covariate for which datapoint is censored
       ca: action taken for censored datapoint
       cy: outcome of censored datapoint (used only if you choose to reveal some censored data!!)
       ntest: size of testset
       xtest: testset covariates
       atest: testset actions
       ytest: testset responses
"""
import numpy as np
import sys
sys.path.insert(1, '../')
import stan_utility


def rbf_test(projectpath, x, y, cx, xtest, ytest):
    """
       Fits rbf model to data and calculates loglikelihood for testset
    """
    bayesname = 'rbf'
    folder = 'rbfmodel/'
    bayespath = projectpath + folder + bayesname + '.stan'
    dat = {'n': len(x),
           'x': x,
           'y': y,
           'cn': len(cx),
           'cx': cx,
           'ntest': len(xtest),
           'xtest': xtest,
           'ytest': ytest
           }
    modelname = bayesname
    model = stan_utility.compile_model(
        bayespath, model_name=modelname, model_path=projectpath+'stancodes/')
    fit = model.sampling(data=dat, seed=194838, chains=4, iter=5000)
    return fit.extract(permuted=True)
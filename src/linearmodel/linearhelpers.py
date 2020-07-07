"""
       Python functions for stan models

       bayesname: name of the model
       stan_folder: folder for compiled stan models
       n: number real datapoints
       x: covariates
       y: responses
       cn: number of censored datapoints
       cx: covariate for which datapoint is censored
       cy: outcome of censored datapoint (used only if you choose to reveal some censored data!!)
       ntest: size of testset
       xtest: testset covariates
       ytest: testset responses
"""
import numpy as np
import sys
sys.path.insert(1, '../')
import stan_utility

def linear_test(projectpath, x, y, cx, xtest, ytest):
    """
       Fits linear model to data and calculates loglikelihood for testset
    """
    bayesname = "bayeslin"
    folder = "linearmodel/"
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
    fit = model.sampling(data=dat, seed=194838, chains=4, iter=2000)
    return fit.extract(permuted=True)


def multi_decision(projectpath, x, y, d, cx, cd, xtest, ytest):
    bayesname = "multidecision_lin"
    folder = "linearmodel/"
    bayespath = projectpath + folder + bayesname + '.stan'
    dat = {'n': len(x),
           'nd': 5, #todo breaks if d doesnt have a sample for each decision
           'd': d,
           'x': x,
           'y': y,
           'cn': len(cx),
           'cx': cx,
           'cd': cd,
           'ntest': len(xtest),
           'xtest': xtest,
           'ytest': ytest
           }
    modelname = bayesname
    model = stan_utility.compile_model(
        bayespath, model_name=modelname, model_path=projectpath+'stancodes/')
    fit = model.sampling(data=dat, seed=194838, chains=4, iter=2000)
    return fit.extract(permuted=True)

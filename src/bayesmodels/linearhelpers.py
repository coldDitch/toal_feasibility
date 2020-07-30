"""
       Python functions for stan models

       bayesname: name of the model
       stan_folder: folder for compiled stan models
       n: number real datapoints
       nd: number of decisions
       d: decisions
       x: covariates
       y: responses
       cn: number of query points
       cx: query covariate
       cd: query decision
       cy: outcome of censored datapoint (used only if you choose to reveal some censored data!!)
       ntest: size of testset
       xtest: testset covariates
       ytest: testset responses
"""
import numpy as np
import sys
sys.path.insert(1, '../')
import stan_utility
import config

### TODO actually no need to have two functions

def to_two_dim(x):
    tx = np.atleast_2d(x)
    tx = tx.reshape(len(x), -1)
    return tx

def format_to_model(projectpath, bayesname, train,  query, test):
    folder = "bayesmodels/"
    bayespath = projectpath + folder + bayesname + '.stan'
    x = to_two_dim(train['x'])
    cx = to_two_dim(query['x'])
    xtest = to_two_dim(test['x'])
    dat = {'n': x.shape[0],
           'k': x.shape[1],
           'nd': test['y'].shape[1],
           'd': train['d'].astype(int),
           'x': x,
           'y': train['y'],
           'cn': len(cx),
           'cx': cx,
           'cd': query['d'],
           'ntest': len(xtest),
           'xtest': xtest,
           'ytest': test['y']
           }
    modelname = bayesname
    model = stan_utility.compile_model(
        bayespath, model_name=modelname, model_path=projectpath+'stancodes/')
    fit = model.sampling(data=dat, seed=194838, chains=4, iter=4000)

    #stan_utility.check_all_diagnostics(fit)
    return fit.extract(permuted=True)

def multi_decision(projectpath, train, query, test):
    return format_to_model(projectpath, 'multidecision_lin', train, query, test)

def gp(projectpath, train, query, test):
    return format_to_model(projectpath, 'gp', train, query, test)
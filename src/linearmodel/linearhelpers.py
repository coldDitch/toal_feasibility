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


def multi_decision(projectpath, train, query, test):
    bayesname = "multidecision_lin"
    folder = "linearmodel/"
    bayespath = projectpath + folder + bayesname + '.stan'
    dat = {'n': len(train['x']),
           'nd': 10, #todo breaks if d doesnt have a sample for each decision
           'd': train['d'],
           'x': train['x'],
           'y': train['y'],
           'cn': len(query['x']),
           'cx': query['x'],
           'cd': query['d'],
           'ntest': len(test['x']),
           'xtest': test['x'],
           'ytest': test['y']
           }
    modelname = bayesname
    model = stan_utility.compile_model(
        bayespath, model_name=modelname, model_path=projectpath+'stancodes/')
    fit = model.sampling(data=dat, seed=194838, chains=4, iter=2000)
    return fit.extract(permuted=True)

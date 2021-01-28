"""
       Python functions for stan models

       bayesname: name of the model
       stan_folder: folder for compiled stan models
       n: number real datapoints
       x: covariates
       y: responses
       cn: number of query points
       cx: query covariate
       cy: outcome of censored datapoint (used only if you choose to reveal some censored data!!)
       ntest: size of testset
       xtest: testset covariates
       ytest: testset responses
"""
import numpy as np
import bayesmodels.stan_utility as stan_utility
import config

def to_two_dim(x):
    """
    Forces covariate matrix to two dimensional matrix even if it is 1 or 0 dimensional.
    """
    if len(x) == 0:
        if config.dataset == 'acic': #TODO the must be a better way to do this
            return np.empty((0, 82))
        else:
            return np.empty((0, config.synthetic_dim))
    tx = np.atleast_2d(x)
    tx = tx.reshape(len(x), -1)
    return tx

def format_to_model(projectpath, bayesname, train, query, test):
    folder = "bayesmodels/"
    bayespath = projectpath + folder + bayesname + '.stan'
    x = to_two_dim(train['x'])
    cx = to_two_dim(query['x'])
    xtest = to_two_dim(test['x'])
    dat = {'n': x.shape[0],
           'k': x.shape[1],
           'nd': config.decision_n,
           'x': x,
           'd': train['d'],
           'y': train['y'],
           'cn': cx.shape[0],
           'cx': cx,
           'cd': query['d'],
           'ntest': xtest.shape[0],
           'xtest': xtest,
           'ytest': test['y']
           }
    modelname = bayesname
    model = stan_utility.compile_model(
        bayespath, model_name=modelname, model_path=projectpath+'stancodes/')
    fit = model.sampling(data=dat, seed=194838, chains=4, iter=1000, verbose=True)
    # length scale has inverse relation to relevancy of covariate
    if config.model == 'ard_se_gp':
        print('lengthscales')
        print(np.mean(fit.extract(permuted=True)['rho'], axis=0))
    if config.run_diagnostics:
        print(fit)
        stan_utility.check_all_diagnostics(fit)
    return fit.extract(permuted=True)

def fit_full(projectpath, train, query, test):
    return format_to_model(projectpath, config.model, train, query, test)

def fit_update(projectpath, train, query, test, x_star, d_star, y_star, samples):
    train_sub = {
        'x': np.append(to_two_dim(train['x']), np.atleast_2d(x_star), axis=0),
        'd': np.append(train['d'], d_star),
        'y': np.append(train['y'], y_star)
    }
    samples = format_to_model(projectpath, config.model, train_sub, query, test)
    return samples

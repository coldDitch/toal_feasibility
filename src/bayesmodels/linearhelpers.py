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

def to_two_dim(x):
    if len(x) == 0:
        if config.dataset == 'acic': #TODO the must be a better way to do this
            return np.empty((0, 82))
        else:
            return np.empty((0, 1))
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
           'x': x,
           'y': train['y'],
           'cn': cx.shape[0],
           'cx': cx,
           'ntest': xtest.shape[0],
           'xtest': xtest,
           'ytest': test['y']
           }
    modelname = bayesname
    model = stan_utility.compile_model(
        bayespath, model_name=modelname, model_path=projectpath+'stancodes/')
    fit = model.sampling(data=dat, seed=194838, chains=4, iter=4000)
    if config.run_diagnostics:
        stan_utility.check_all_diagnostics(fit)
    return fit.extract(permuted=True)

def fit_full(projectpath, train, query, test):
    sample_col = {
        'u_bar': np.empty((8000, test['y'].shape[0], test['y'].shape[1])),
        'py': np.empty((8000, query['y'].shape[0]))
    }
    for d in range(1,config.decision_n+1):
        train_sub = {
            'x': train['x'][d==train['d']],
            'y': train['y'][d==train['d']],
        }
        query_sub = {
            'x': query['x'][d==query['d']],
        }
        test_sub = {
            'x': test['x'],
            'y': test['y'][:,d-1]
        }
        samples = format_to_model(projectpath, config.model, train_sub, query_sub, test_sub)
        sample_col['u_bar'][:,:,d-1] = samples['u_bar']
        if np.any(d==query['d']):
            sample_col['py'][:,d==query['d']] = samples['py']
    return sample_col

def fit_update(projectpath, train, query, test, x_star, d_star, y_star, samples):
    train_sub = {
        'x': np.append(to_two_dim(train['x'][d_star==train['d']]), np.atleast_2d(x_star), axis=0),
        'y': np.append(train['y'][d_star==train['d']], y_star)
    }
    test_sub = {
        'x': test['x'],
        'y': test['y'][:,d_star-1]
    }
    samples_sub = format_to_model(projectpath, config.model, train_sub, query, test_sub)
    new_samples = {}
    new_samples['u_bar'] = np.copy(samples['u_bar'])
    new_samples['u_bar'][:, :, d_star-1] = samples_sub['u_bar']
    return new_samples



def multi_decision(projectpath, train, query, test):
    return format_to_model(projectpath, 'linear', train, query, test)

def gp(projectpath, train, query, test):
    return format_to_model(projectpath, 'gp', train, query, test)

import random
import math
import numpy as np
import pandas as pd
import numpy.random as rndm
import matplotlib.pyplot as plt
from scipy.stats import norm
from bayesmodels import linearhelpers
import config

def generate_dataset(problem, training_size, test_size, query_size, decision_n, seed):
    if problem == 'acic':
        return generate_acic_dataset(problem, training_size, test_size, query_size, decision_n, seed)
    elif problem == 'synthetic':
        return generate_multidecision_dataset(problem, training_size, test_size, query_size, decision_n, seed)
    else:
        print("CHOOSE PROPER DATASET")


def is_categorical(val):
    return val.dtype==np.dtype('O')

def to_binary(val):
    uniques = np.unique(val)
    mat = np.zeros((len(val),len(uniques)))
    for i in range(len(uniques)):
        byte_arr = (val == uniques[i]).astype('b')
        mat[:, i] = byte_arr
    return mat

def covariate_matrix():
    df = pd.read_csv(config.acic_path + 'x.csv')
    col = []
    for key in df.keys():
        covariate = df[key].values
        if is_categorical(covariate):
            col.append(to_binary(covariate))
        else:
            col.append(covariate.reshape(-1,1))
    col = np.concatenate(col,axis=1)
    return col

def normalize(mat):
    mean = np.mean(mat, axis=0)
    l = len(mean) if type(mean) is np.ndarray else 1
    mat = mat - mean
    std = np.array(np.std(mat, axis=0))
    std = np.max((std,np.ones(l)))
    mat = mat / std
    return mat, mean, std

def acic_covariates():
    mat = covariate_matrix()
    mat, _, _ = normalize(mat)
    return mat

def acic_labels(fil):
    df = pd.read_csv(config.acic_path + fil)
    potential_outcomes = df[['mu0', 'mu1']].values
    treatments = df['z'].values
    #treatments = np.random.randint(1, size=potential_outcomes.shape[0]) 
    outcomes = np.array([potential_outcomes[i, treatments[i]] for i in range(len(treatments))])
    treatments = treatments + 1
    return outcomes, treatments, potential_outcomes

def generate_acic_dataset(p, training_size, test_size, query_size, decision_n, seed):
    np.random.seed(seed)
    num_queries = query_size
    num_files = int(math.ceil(decision_n / 2))
    covariates = acic_covariates()
    num_samples = len(covariates)
    outcomes = []
    treatments = []
    potential_outcomes = []
    for i in range(num_files):
        outcome, treatment, potential_outcome = acic_labels(config.acic_files[i])
        low = int(num_samples * i / num_files)
        high = int(num_samples * (i+1) / num_files)
        outcomes.append(outcome[low:high])
        treatments.append(treatment[low:high]+i*2)
        potential_outcomes.append(potential_outcome)
    outcomes = np.concatenate(outcomes, axis=0)
    treatments = np.concatenate(treatments, axis=0)
    potential_outcomes = np.concatenate(potential_outcomes, axis=1)
    indexes = np.random.choice(len(treatments), training_size + test_size + num_queries, replace=False)
    ind_train = indexes[:training_size]
    ind_test = indexes[training_size:training_size+test_size]
    ind_query = indexes[training_size+test_size:num_queries+training_size+test_size]

    # normalize using the known training outcomes
    outcomes[ind_train], mean, std = normalize(outcomes[ind_train])
    outcomes[ind_test] = (outcomes[ind_test] - mean) / std
    outcomes[ind_query] = (outcomes[ind_query] - mean) / std

    train = {
        'x': covariates[ind_train],
        'd': treatments[ind_train],
        'y': outcomes[ind_train]
    }
    test = {
        'x': covariates[ind_test],
        'y': potential_outcomes[ind_test]
    }
    query = {
        'x': covariates[ind_query],
        'd': treatments[ind_query],
        'y': outcomes[ind_query]
    }
    revealed = {
        'x': np.empty(0),
        'd': np.empty(0),
        'y': np.empty(0)
    }
    return train, query, test, revealed


def generate_params(seed):
    np.random.seed(seed)
    coef_1 = np.random.uniform(-20, 20)
    coef_2 = np.random.uniform(-20, 20)
    return coef_1, coef_2 

def generate_multidecision_dataset(problem, training_size, test_size, query_size, decision_n, seed):
    num_queries = query_size
    std = 1
    # query set from which model chooses x and d, for which we reveal y
    query_x = covariate_dist(num_queries)
    query_d = np.random.randint(1, decision_n+1, num_queries)
    query_y = np.zeros(num_queries)
    # initial data for training 
    train_x = covariate_dist(training_size)
    train_d = np.random.randint(1, decision_n+1, training_size)
    train_y = np.zeros(training_size)
    # test set, for test set outcome for all decisions are known to find the best decision
    test_x = covariate_dist(test_size)
    test_y = np.zeros((test_size, decision_n))
    for i in range(1, decision_n + 1):
        slope, intercept = generate_params(seed+i)
        query_y[query_d==i] = slope * query_x[query_d==i] + intercept + np.random.normal(0, std, len(query_x[query_d==i]))
        train_y[train_d==i] = slope * train_x[train_d==i] + intercept + np.random.normal(0, std, len(train_x[train_d==i]))
        test_y[:,i-1] = slope * test_x + intercept + np.random.normal(0, std, test_size)
    train = {
        'x': train_x,
        'y': train_y,
        'd': train_d
        }
    query = {
        'x': query_x, 
        'y': query_y,
        'd': query_d
    }
    test = {
        'x': test_x,
        'y': test_y
    }
    revealed = {
        'x': np.empty(0),
        'd': np.empty(0),
        'y': np.empty(0)
    }
    sort_by_covariates(test)
    sort_by_covariates(query)
    sort_by_covariates(test)
    return train, query, test, revealed


def sort_by_covariates(dat):
    sort_index = np.argsort(dat['x'])
    for d in dat.keys():
        dat[d] = dat[d][sort_index]

def covariate_dist(N):
    return np.random.random(N)*9 - 4.5

def choose_fit(problem):
    if problem == 'linear':
        return linearhelpers.fit_full
    elif problem == 'gp':
        return linearhelpers.gp
    else:
        print('CHOOSE PROPER PROBLEM')

def mean_conf(dat):
    # mean and it's standard deviation
    N, Nq = dat.shape
    ret = np.empty((3, Nq))
    ret[0, :] = np.mean(dat, axis=0)
    ret[1, :] = ret[0, :] - np.std(dat, axis=0)/np.sqrt(N)
    ret[2, :] = ret[0, :] + np.std(dat, axis=0)/np.sqrt(N)
    return ret

def shadedplot(x, y, fill=True, label='', color='b'):
    # y[0,:] mean, median etc; in the middle
    # y[1,:] lower
    # y[2,:] upper
    p = plt.plot(x, y[0, :], label=label, color=color)
    c = p[-1].get_color()
    if fill:
        plt.fill_between(x, y[1, :], y[2, :], color=c, alpha=0.25)


def plot_run(samples, test, train, revealed, run_name, plot):
    if not plot:
        return
    print("Plotting")
    test['x'] = test['x'].reshape(test['x'].shape[0], -1)
    train['x'] = train['x'].reshape(train['x'].shape[0], -1)
    for cov in range(test['x'].shape[1]):
        marg_dat = {
        'x': train['x'][:,cov],
        'y': train['y'],
        'd': train['d']
        }
        sort_by_covariates(marg_dat)
        decisions = config.decision_n
        np.random.seed(1234)
        for decision in range(decisions):
            res = np.empty((3, test['x'].shape[0]))
            mu = samples["mu_test"][:,:,decision]
            plot_dat = {
                'x': test['x'][:,cov],
                'mu': mu.T
            }
            sort_by_covariates(plot_dat)
            res[0] = np.mean(plot_dat['mu'].T, axis=0)
            res[1] = res[0]+np.std(plot_dat['mu'].T, axis=0)
            res[2] = res[0]-np.std(plot_dat['mu'].T, axis=0)
            color = np.random.rand(3,)
            shadedplot(plot_dat['x'], res, color=color, label='prediction d='+str(decision))
            if revealed['x'].shape[0] > 0:
                revealed['x'] = revealed['x'].reshape(revealed['x'].shape[0], -1)
                rev_ind = [revealed['d']==decision+1]
                plt.scatter(revealed['x'][:,cov][rev_ind], revealed['y'][rev_ind], color=color, label='query d='+str(decision))
            else:
                #plt.scatter(test['x'], test['y'][:,decision])
                plt.scatter(marg_dat['x'][decision+1==marg_dat['d']], marg_dat['y'][decision+1==marg_dat['d']], color=color)
        plt.legend()
        plt.savefig('./plots/'+run_name+'.png')
        plt.show()
        plt.clf()

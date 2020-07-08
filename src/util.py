import numpy as np
import numpy.random as rndm
from linearmodel import linearhelpers
import matplotlib.pyplot as plt
from scipy.stats import norm


def generate_params(seed):
    np.random.seed(seed)
    coef_1 = np.random.uniform(-20, 20)
    coef_2 = np.random.uniform(-20, 20)
    return coef_1, coef_2 

def generate_multidecision_dataset(problem, training_size, test_size, seed):
    num_decisions = 10
    num_queries = 10
    std = 1
    # query set from which model chooses x and d, for which we reveal y
    query_x = covariate_dist(num_queries)
    query_d = np.random.randint(1, num_decisions+1, num_queries)
    query_y = np.zeros(num_queries)
    # initial data for training 
    train_x = covariate_dist(training_size)
    train_d = np.random.randint(1, num_decisions+1, training_size)
    train_y = np.zeros(training_size)
    # test set, for test set outcome for all decisions are known to find the best decision
    test_x = covariate_dist(test_size)
    test_y = np.zeros((test_size, num_decisions))
    for i in range(1, num_decisions + 1):
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
    if problem == 'multilin':
        return linearhelpers.multi_decision
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


def plot_run(samples, test, revealed, run_name, plot):
    if not plot:
        return
    print("Plotting")
    decisions = int(samples['num_decisions'][0])
    np.random.seed(1234)
    for decision in range(decisions):
        res = np.empty((3, test['x'].shape[0]))
        mu = samples["mu_test"][:,:,decision]
        res[0] = np.mean(mu, axis=0)
        res[1] = res[0]+np.std(mu, axis=0)
        res[2] = res[0]-np.std(mu, axis=0)
        color = np.random.rand(3,)
        shadedplot(test['x'], res, color=color, label='prediction d='+str(decision))
        if revealed['x'].shape[0] > 0:
            rev_ind = [revealed['d']==decision+1]
            plt.scatter(revealed['x'][rev_ind], revealed['y'][rev_ind], color=color, label='query d='+str(decision))
    plt.legend()
    plt.savefig('./plots/'+run_name+'.png')
    plt.show()
    plt.clf()

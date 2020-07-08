import numpy as np
import numpy.random as rndm
from linearmodel import linearhelpers
from rbfmodel import rbfhelpers
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
    query_x = covariate_dist(num_queries, 'uniform')
    query_d = np.random.randint(1, num_decisions+1, num_queries)
    query_y = np.zeros(num_queries)
    train_x = covariate_dist(training_size, 'uniform')
    train_d = np.random.randint(1, num_decisions+1, training_size)
    train_y = np.zeros(training_size)
    test_x = covariate_dist(test_size, 'uniform')
    test_y = np.zeros((test_size, num_decisions))
    for i in range(1, num_decisions + 1):
        slope, intercept = generate_params(seed+i)
        query_y[query_d==i] = slope * query_x[query_d==i] + intercept + np.random.normal(0, std, len(query_x[query_d==i]))
        train_y[train_d==i] = slope * train_x[train_d==i] + intercept + np.random.normal(0, std, len(train_x[train_d==i]))
        test_y[:,i-1] = slope * test_x + intercept + np.random.normal(0, std, test_size)

    train = (train_x, train_y, train_d)
    query = (query_x, query_y, query_d)
    test = (test_x, test_y)
        
    return (train, query), test

def generate_datasets(problem, training_size, test_size, seed):
    if problem == "linear":
        slope, intercept = generate_params(seed)
        train = generate_linear(slope, intercept, N=training_size)
        query = generate_linear(slope, intercept, N=10)
        test = generate_linear(slope, intercept, N=test_size)
    elif problem == "test_case":
        # adhoc stuff for debugging
        radial_1, radial_2 = generate_params(seed)
        train = generate_rbf(radial_1, radial_2, N=training_size, x_dist='debug_train')
        test = generate_rbf(radial_1, radial_2, N=test_size, x_dist='debug_test')
        x_que = np.array([-2, 2])
        y_que = compute_theta([8, -20], x_que)
        query = (x_que, y_que)
    else:
        # radial basis function for testing on rbf or gp models
        radial_1, radial_2 = generate_params(seed)
        train = generate_rbf(radial_1, radial_2, N=training_size, x_dist='beta')
        test = generate_rbf(radial_1, radial_2, N=test_size, x_dist='beta')
        query = generate_rbf(radial_1, radial_2, N=10, x_dist='range')
    return (train, query), test


def choose_fit(problem):
    #choose the stanmodel, returns the posterior samples
    if problem == "linear":
        return linearhelpers.linear_test
    elif problem == "rbf":
        return rbfhelpers.rbf_test
    elif problem == "test_case":
        return rbfhelpers.rbf_test
    elif problem == 'multilin':
        return linearhelpers.multi_decision
    else:
        print("Problem not specified correctly")
        return


def rbf(x, i):
    c = np.array([-2., 2.])
    ret = np.exp(-(x - c[i])**2)
    return ret


def compute_theta(beta, x):
    num_centers = 2
    return np.sum([beta[i]*rbf(x, i) for i in range(0, num_centers)], 0)


def covariate_dist(N, x_dist):
    # different imbalanced covariate distributions for different tasks
    if x_dist == 'beta':
        x = np.random.beta(1, 2, size=N)*6 - 3
    elif x_dist == 'range':
        x = np.arange(-3.1, 3.0, 6.0 / (N-1))
    elif x_dist == 'debug_test':
        x = np.random.random(N-1)*3 - 3
        x = np.append(x, [2])
    elif x_dist == 'debug_train':
        x = np.random.random(N)*4 - 3
    elif x_dist == 'uniform':
        x = np.random.random(N)*9 - 4.5
    else:
        print("Please specify the distribution for covariates")
        return
    return x


def generate_rbf(radial_1, radial_2, N=30, x_dist='uniform', std=5):
    x = covariate_dist(N, x_dist)
    beta = np.array([8, -30])
    y = compute_theta(beta, x) + np.random.normal(0, std, N)
    return (x, y)


def generate_linear(slope, intercept, N=30, x_dist='uniform', std=2):
    x = covariate_dist(N, x_dist)
    y = slope * x + intercept + np.random.normal(0, std, N)
    return (x, y)


def bboot(N, B=10000):
    '''
    gives bayesian bootstrapping weights
    '''
    g = rndm.gamma(np.ones((N, B)))
    g /= np.sum(g, 0)[None, :]
    return g


def bootstrap_results(dat, prctile=[5, 95]):
    '''
    Input: Data of one elicitation (N x Nq)
    Returns returns means and errorbounds.
    Bayesian bootstrapping is used for the errorbounds.
    Assuming we have N observations and Nq queries,
    the function returns 3 x Nq numpy array with mean (idx 0)
    and percentiles (idxs 1 and 2 )
    '''
    N, Nq = dat.shape
    w = bboot(N)
    ret = np.empty((3, Nq))
    ret[0, :] = np.mean(dat, axis=0)
    means = np.dot(w.T, dat)
    ret[1, :] = np.percentile(means, prctile[0], axis=0)
    ret[2, :] = np.percentile(means, prctile[1], axis=0)
    return ret

def mean_conf(dat):
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
    #plt.plot(x, y[1,:], color=c, alpha=0.25)
    #plt.plot(x, y[2,:], color=c, alpha=0.25)
    if fill:
        plt.fill_between(x, y[1, :], y[2, :], color=c, alpha=0.25)

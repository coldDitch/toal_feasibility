import sys
import stan_utility
import config
import pickle
import numpy as np
import random
from sklearn.neighbors import KernelDensity
from util import choose_fit, generate_multidecision_dataset, shadedplot, plot_run, generate_dataset
from bayesmodels.linearhelpers import fit_full, fit_update
import matplotlib
import matplotlib.pyplot as plt
import timeit


def random_sampling(samples, data):
    #acquistion function which chooses next query randomly
    return random.randint(0, len(data['query']['x'])-1)


def uncertainty_sampling_y(samples, data):
    #acquisition function which chooses next query based on largest uncertainty
    var = np.var(samples['py'], axis=0)
    return np.argmax(var)

def decision_ig(samples, data):
    #acquisition which minimizes entropy of 
    return toal(samples, data, 'decision_ig', entropy_of_maximizer_decision)

def eig(samples, data):
    return toal(samples, data, 'eig', estimate_entropy_1D)


def estimate_bandwidth(samples):
    # approximation for minimizing integrated mse
    return np.min((np.std(samples), (np.quantile(samples, 0.75)-np.quantile(samples, 0.25))/1.34)) * 0.9 * np.power(len(samples), -0.2)


def estimate_entropy_1D(sampledata):
    # estimates entropy of dataset where dimension 1 are samples from the distribution and
    # dimension 2 for different samples
    samples = sampledata['u_bar']
    ntarget = samples.shape[1]
    entropy = 0
    for i in range(ntarget):
        for d in range(config.decision_n):
            # approximate bandwidth for minimizing the mean integrated squared error
            bandwidth = estimate_bandwidth(samples[:, i, d])
            # kernel density estimation and then Monte Carlo estimate of entropy
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(
                samples[:, i].reshape(-1, 1))
            # number of samples determines the accuracy of estimate of one dimensional entropy
            y = kde.sample(50)
            entropy -= np.mean(kde.score_samples(y))
            # plot kde for debugging
            if config.plot_run:
                X_plot = np.linspace(np.min(samples[:, i, d]), np.max(
                    samples[:, i, d]), 50)[:, np.newaxis]
                log_dens = kde.score_samples(X_plot)
                plt.plot(X_plot[:, 0], np.exp(log_dens), '-',
                        label="Gaussian kernel")
                plt.legend(loc='upper left')
                plt.plot(samples[:, i, d], -0.005 - 0.01 *
                        np.random.random(samples.shape[0]), '+k')
                plt.show()
    return(np.mean(entropy))


def toal(samples, data, objective_utility, entropy_fun):
    def f(i):
        print("ITER")
        print(i)
        print(data['query']['x'][i].shape)
        # Gauss-Hermite quadrature to compute the integral
        points, weights = np.polynomial.hermite.hermgauss(
            32)  # should be atleast 32
        if config.plot_run:
            y_stars = []
            entropies = []
        expected_entropy = 0
        for ii, yy in enumerate(points):
            # predicted mean and standard deviation of point x
            mu_x, sd_x = np.mean(samples['py'][i]), np.std(samples['py'][i])
            y_star = np.sqrt(2)*sd_x*yy + mu_x  # for substitution

            # create new training set for the model
            # fit model again
            samples_new = fit_update(data['projectpath'], data['train'], data['query'], data['test'], data['query']['x'][i], data['query']['d'][i], y_star, samples)
            ### Expected utility debug
            #
            #revealed = {
            #    'x': np.array([data['query']['x'][i]]),
            #    'd': np.array([data['query']['d'][i]]),
            #    'y': np.array([y_star])
            #}
            #plot_run(samples_new, data['test'], data['train'], revealed, 'toal_imputation', config.plot_run)
            ###
            H = entropy_fun(samples_new)
            expected_entropy += H * weights[ii] * 1/np.sqrt(np.pi)
            if config.plot_run:
                entropies.append(H)
                y_stars.append(y_star)
        if config.plot_run:
            plt.plot(y_stars, entropies)
            plt.xlabel('y_star')
            plt.ylabel('H(D_best)')
            plt.savefig('./plots/expectation.png')
            plt.show()
        return expected_entropy

    expected_utils = [f(i) for i in range(data['query']['x'].shape[0])]
    i_star = np.argmin(expected_utils)
    x_star = data['query']['x'][i_star]
    if config.plot_run:
        print("MINIMUM")
        print(x_star)
        print("ENT")
        print(expected_utils[i_star])
        for d in range(1, config.decision_n+1):
            color = np.random.rand(3,)
            plt.scatter(data['query']['x'][data['query']['d']==d], np.array(expected_utils)[data['query']['d']==d],color=color, label='d='+str(d-1))
        plt.legend()
        plt.title('h(' + objective_utility+')')
        plt.xlabel('query points')
        plt.ylabel('expected entropy of decisions')
        plt.savefig('./plots/entropies.png')
        plt.show()
        plt.clf()
    return i_star

def entropy_of_maximizer_decision(sampledata):
    decisions = config.decision_n
    samples = sampledata["u_bar"]
    entropies = []
    num_data = samples.shape[1]
    for i in range(num_data):
        entropy = 0
        for decision in range(decisions):
            maximizers = np.ones(samples.shape[0], dtype=bool)
            for d in range(decisions):
                maximizers = np.logical_and((samples[:, i, d] <= samples[:, i, decision]), maximizers)
            prob = np.sum(maximizers) / len(maximizers)
            if prob != 0:
                entropy -= prob * np.log(prob)
        entropies.append(entropy)
    return np.mean(entropies)


def choose_criterion(criterion):
    # choose acquisition criterion which returns the index for the next acquisition
    if criterion == "random":
        return random_sampling
    elif criterion == "uncer_y":
        return uncertainty_sampling_y
    elif criterion == "decision_ig":
        return decision_ig
    elif criterion == "eig":
        return eig
    else:
        print("Activelearning not specified correctly")
        return


def decision_acc(samples, test):
    correct_count = 0
    for i in range(test['y'].shape[0]):
        best_decision = 0
        model_decision = 0
        decision_util = -np.infty
        model_util = -np.infty
        for j in range(test['y'].shape[1]):
            if test['y'][i, j] > decision_util:
                decision_util = test['y'][i, j]
                best_decision = j
            mu_util = np.mean(samples['u_bar'][:, i, j])
            if mu_util > model_util:
                model_util = mu_util
                model_decision = j
        if model_decision == best_decision:
            correct_count += 1
    return correct_count / test['y'].shape[0]


def save_data(dat_save, samples, test):
    print("SAVING")
    #dat_save["logl"].append(np.mean(np.exp(samples['logl'])))
    dat_save["acc"].append(decision_acc(samples, test))
    dat_save["dent"].append(entropy_of_maximizer_decision(samples))


def active_learning(projectpath, seed, criterion, steps):
    active_learning_func = choose_criterion(criterion)
    decision_n = config.decision_n
    training_size = config.train_n
    test_size = config.test_n
    query_size = config.query_n
    problem = config.dataset
    np.random.seed(seed)
    variables = ['x', 'd', 'y']
    run_name = problem + '-' + criterion + "-" + \
        str(training_size) + "-" + str(test_size) + \
        "-" + str(steps) + "-" + str(seed)
    train, query, test, revealed = generate_dataset(problem, training_size, test_size, query_size, decision_n, seed)
    # true probability of censoring
    print("missing shape")
    print(query['x'].shape)
    print("observed shape")
    print(train['x'].shape)
    dat_save = {
        "logl": [],
        "acc": [],
        "dent": []
    }
    samples = fit_full(projectpath, train, query, test)
    plot_run(samples, test, train, revealed, run_name+'-0', config.plot_run)
    save_data(dat_save, samples, test)
    for iteration in range(steps):
        data = {'projectpath': projectpath,
                'train': train,
                'query': query,
                'test': test
        }
        new_ind = active_learning_func(samples, data)
        print("Iteration " + str(iteration) + ". Acquire point at index " +
              str(new_ind) + ": x=" + str(query['x'][new_ind]))
        print("train", train['x'].shape)
        print("query", query['x'][new_ind].shape)
        for v in variables:
            if type(query[v][new_ind]) is np.ndarray:
                train[v] = np.append(train[v], np.atleast_2d(query[v][new_ind]), axis=0)
            else:
                train[v] = np.append(train[v], query[v][new_ind])
            revealed[v] = np.append(revealed[v], query[v][new_ind])
            query[v] = np.delete(query[v], new_ind, axis=0)
        samples = fit_full(projectpath, train, query, test)
        print("train", train['x'].shape)
        print("query", query['x'].shape)
        save_data(dat_save, samples, test)
        plot_run(samples, test, train, revealed, run_name+'-'+str(iteration+1), config.plot_run)
    print(dat_save)
    print(train['d'])
    dat_save['querydvals'] = revealed['d']
    dat_save['queryxvals'] = revealed['x']
    dat_save['queryyvals'] = revealed['y']
    filename = projectpath + "res/" + run_name
    pickle.dump(dat_save, open(filename + ".p", "wb"))


def main():
    projectpath = sys.argv[1]
    seed = int(sys.argv[2])
    criterion = sys.argv[3]
    active_learning_steps = int(sys.argv[4])
    active_learning(projectpath, seed, criterion, active_learning_steps)


if __name__ == "__main__":
    main()

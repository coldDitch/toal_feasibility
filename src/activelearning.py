"""
Main file, implements different active learning functions for decision making
"""
import sys
import random
import pickle
import numpy as np
from sklearn.neighbors import KernelDensity
from data_preprocess import generate_dataset
from plot_utils import plot_all, plot_density_estimate, plot_expected_values, plot_query_fun, estimate_bandwidth
from bayesmodels.model_training_helpers import fit_full, fit_update
import config


def random_sampling(samples, data):
    """
    acquistion function which chooses next query randomly
    """
    return random.randint(0, len(data["query"]["x"])-1)


def uncertainty_sampling_y(samples, data):
    """
    acquisition function which chooses next query based on largest uncertainty
    """
    var = np.var(samples["py"], axis=0)
    return np.argmax(var)

def decision_ig(samples, data):
    """
    acquisition which minimizes entropy of decisions
    """
    return expected_value_minimizer(samples, data, "decision_ig", entropy_of_maximizer_decision)

def eig(samples, data):
    """
    acquisition which minimizes entropy of average utility
    """
    return expected_value_minimizer(samples, data, "eig", estimate_entropy_1D)


def estimate_entropy_1D(sampledata):
    """
    sum of entropies of 1 dimensional utilities over samples and decisions
    """
    samples = sampledata["u_bar"]
    ntarget = samples.shape[1]
    entropy = 0
    for i in range(ntarget):
        for d in range(config.decision_n):
            # approximate bandwidth for minimizing the mean integrated squared error
            bandwidth = estimate_bandwidth(samples[:, i, d])
            # kernel density estimation and then Monte Carlo estimate of entropy
            kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
                samples[:, i, d].reshape(-1, 1))
            # number of samples determines the accuracy of estimate of 
            # one dimensional entropy, increase this if eig doesnt perform well
            y = kde.sample(500)
            entropy -= np.mean(kde.score_samples(y))
            # plot kde for debugging
            plot_density_estimate(samples, kde, i, d)
    return(np.mean(entropy))


def expected_value_minimizer(samples, data, objective_utility, entropy_fun):
    """
    finds index i for query x_i, d_i in query set which minimizes the entropy_fun
    """
    def expected_entropy_fun(i):
        """ 
        evaluates expected value for entropy_fun
        """
        # Gauss-Hermite quadrature to compute the integral,
        # number of points should be around 30, depends how smooth u is as function of y^*
        points, weights = np.polynomial.hermite.hermgauss(
            32)
        if config.save_plots or config.show_plots:
            y_stars = []
            entropies = []
        expected_entropy = 0
        for ii, yy in enumerate(points):
            # predicted mean and standard deviation of point x
            mu_x, sd_x = np.mean(samples["py"][i]), np.std(samples["py"][i])
            y_star = np.sqrt(2)*sd_x*yy + mu_x
            # fit model again
            samples_new = fit_update(data["projectpath"], data["train"], data["query"],\
                data["test"], data["query"]["x"][i], data["query"]["d"][i], y_star, samples)
            H = entropy_fun(samples_new)
            expected_entropy += H * weights[ii] * 1/np.sqrt(np.pi)
            if config.show_plots or config.save_plots:
                entropies.append(H)
                y_stars.append(y_star)
        if config.save_plots or config.show_plots:
            plot_query_fun(y_stars, entropies)
        return expected_entropy
    expected_utils = [expected_entropy_fun(i) for i in range(data["query"]["x"].shape[0])]
    i_star = np.argmin(expected_utils)
    if config.save_plots or config.show_plots:
        plot_expected_values(data, expected_utils, objective_utility)
    return i_star

def entropy_of_maximizer_decision(sampledata):
    """
    entropy of best decisions H(D_best)
    """
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
    """
    choose acquisition criterion which returns the index for the next acquisition
    """
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
    """
    accuracy of decisions, the actual best decisions is compared to best decisions provided by model
    """
    correct_count = 0
    for i in range(test["y"].shape[0]):
        best_decision = 0
        model_decision = 0
        decision_util = -np.infty
        model_util = -np.infty
        for j in range(test["y"].shape[1]):
            if test["y"][i, j] > decision_util:
                decision_util = test["y"][i, j]
                best_decision = j
            mu_util = np.mean(samples["u_bar"][:, i, j])
            if mu_util > model_util:
                model_util = mu_util
                model_decision = j
        if model_decision == best_decision:
            correct_count += 1
    return correct_count / test["y"].shape[0]


def save_data(dat_save, samples, test):
    """
    save data for future plots
    """
    dat_save["acc"].append(decision_acc(samples, test))
    dat_save["dent"].append(entropy_of_maximizer_decision(samples))


def active_learning(projectpath, seed, criterion, steps):
    """
    main function which calls for query updates model
    and evaluates the model after each query
    """
    active_learning_func = choose_criterion(criterion)
    np.random.seed(seed)
    variables = ["x", "d", "y"]
    run_name = config.dataset + "-" + config.model + "-" + criterion + "-" + \
        str(config.train_n) + "-" + str(config.test_n) + "-" + str(config.decision_n) + \
        "-" + str(config.query_n) + "-" + str(steps) + "-" + str(seed)
    train, query, test, revealed = generate_dataset(seed)
    # true probability of censoring
    print("query shape")
    print(query["x"].shape)
    print("observed shape")
    print(train["x"].shape)
    dat_save = {
        "logl": [],
        "acc": [],
        "dent": []
    }
    samples = fit_full(projectpath, train, query, test)
    plot_all(samples, test, train, run_name+"-0")
    save_data(dat_save, samples, test)
    for iteration in range(steps):
        data = {"projectpath": projectpath,
                "train": train,
                "query": query,
                "test": test
        }
        new_ind = active_learning_func(samples, data)
        print("Iteration " + str(iteration) + ". Acquire point at index " +
              str(new_ind) + ": x=" + str(query["x"][new_ind]))
        print("train", train["x"].shape)
        print("query", query["x"][new_ind].shape)
        for v in variables:
            if type(query[v][new_ind]) is np.ndarray:
                train[v] = np.append(train[v], np.atleast_2d(query[v][new_ind]), axis=0)
            else:
                train[v] = np.append(train[v], query[v][new_ind])
            revealed[v] = np.append(revealed[v], query[v][new_ind])
            query[v] = np.delete(query[v], new_ind, axis=0)
        samples = fit_full(projectpath, train, query, test)
        print("train", train["x"].shape)
        print("query", query["x"].shape)
        save_data(dat_save, samples, test)
        plot_all(samples, test, train, run_name+"-"+str(iteration+1))
    print(dat_save)
    print(train["d"])
    dat_save["querydvals"] = revealed["d"]
    dat_save["queryxvals"] = revealed["x"]
    dat_save["queryyvals"] = revealed["y"]
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

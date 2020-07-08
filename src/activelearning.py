import sys
import pickle
import numpy as np
import random
from util import choose_fit, generate_multidecision_dataset, shadedplot, plot_run
import matplotlib
import matplotlib.pyplot as plt

PLOT_DATA_AND_MODEL = False
PLOT_EXPECTED_ENTROPY = False


def random_sampling(samples, fit_model, data):
    #acquistion function which chooses next query randomly
    return random.randint(0, len(data['query']['x'])-1)


def uncertainty_sampling_y(samples, fit_model, data):
    #acquisition function which chooses next query based on largest uncertainty
    var = np.var(samples['py'], axis=0)
    return np.argmax(var)

def decision_ig(samples, fit_model, data):
    #acquisition which minimizes entropy of 
    return toal(samples, fit_model, data, 'decision_ig', entropy_of_maximizer_decision)

def toal(samples, fit_model, data, objective_utility, entropy_fun):

    def f(x):
        i = np.where(np.isclose(data['query']['x'], x).reshape(-1))[0][0]
        expected_entropy = 0
        # Gauss-Hermite quadrature to compute the integral
        points, weights = np.polynomial.hermite.hermgauss(
            32)  # should be atleast 32
        print("QUERY COV")
        print(data["query"]['x'][i])
        for ii, yy in enumerate(points):
            # predicted mean and standard deviation of point x
            mu_x, sd_x = np.mean(samples['py'][i]), np.std(samples['py'][i])
            y_star = np.sqrt(2)*sd_x*yy + mu_x  # for substitution

            # create new training set for the model
            train = {
                'x': np.append(data['train']['x'], data['query']['x'][i]),
                'd': np.append(data['train']['d'], data['query']['d'][i]),
                'y': np.append(data['train']['y'], y_star)
            } 
            # fit model again
            samples_new = fit_model(data['projectpath'], train, data['query'], data['test'])
            H = entropy_fun(samples_new, objective_utility)
            expected_entropy += H * weights[ii] * 1/np.sqrt(np.pi)
        return expected_entropy

    # evaluate all possible query points
    expected_utils = [f(x) for x in data['query']['x']]
    plt.plot(data['query']['x'], expected_utils)
    i_star = np.argmin(expected_utils)
    x_star = data['query']['x'][i_star]
    if PLOT_EXPECTED_ENTROPY:
        print("MINIMUM")
        print(x_star)
        plt.title('h(' + objective_utility+')')
        plt.xlabel('queries')
        plt.ylabel('entropy of expected utility')
        plt.plot(data['query']['x'], expected_utils)
        plt.show()
        plt.clf()
    return i_star

def entropy_of_maximizer_decision(sampledata, name):
    decisions = int(sampledata["num_decisions"][0])
    samples = sampledata["mu_test"]
    print(samples.shape)
    entropies = []
    num_samples = samples.shape[1]
    for i in range(num_samples):
        entropy = 0
        for decision in range(decisions):
            prob = 1 / num_samples**decisions
            for d in range(decisions):
                prob *= np.sum(samples[:, i, d] <= samples[:, i, decision])
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
            mu_util = np.mean(samples['mu_test'][:, i, j])
            if mu_util > model_util:
                model_util = mu_util
                model_decision = j
        if model_decision == best_decision:
            correct_count += 1
    return correct_count / test['y'].shape[0]

def save_data(dat_save, samples, test):
    print("SAVING")
    dat_save["logl"].append(np.mean(np.exp(samples['logl'])))
    dat_save["acc"].append(decision_acc(samples, test))


def active_learning(problem, training_size, test_size, projectpath, seed, active_learning_func, steps, fit_model, criterion):
    np.random.seed(seed)
    variables = ['x', 'd', 'y']
    run_name = problem + '-' + criterion + "-" + \
        str(training_size) + "-" + str(test_size) + \
        "-" + str(steps) + "-" + str(seed)
    train, query, test, revealed = generate_multidecision_dataset(problem, training_size, test_size, seed)

    # true probability of censoring
    print("missing shape")
    print(query['x'].shape)
    print("observed shape")
    print(train['x'].shape)
    dat_save = {
        "logl": [],
        "acc": [],
    }
    samples = fit_model(projectpath, train, query, test)
    plot_run(samples, test, revealed, run_name+'-0', PLOT_DATA_AND_MODEL)
    save_data(dat_save, samples, test)
    for iteration in range(steps):
        data = {'projectpath': projectpath,
                'train': train,
                'query': query,
                'test': test
        }
        new_ind = active_learning_func(samples, fit_model, data)
        print("Iteration " + str(iteration) + ". Acquire point at index " +
              str(new_ind) + ": x=" + str(query['x'][new_ind]))
        for v in variables:
            train[v] = np.append(train[v], query[v][new_ind])
            revealed[v] = np.append(revealed[v], query[v][new_ind])
            query[v] = np.delete(query[v], new_ind)
        samples = fit_model(projectpath, train, query, test)
        save_data(dat_save, samples, test)
        plot_run(samples, test, revealed, run_name+'-'+str(iteration), PLOT_DATA_AND_MODEL)
    print(dat_save)
    dat_save['querydvals'] = revealed['d']
    dat_save['queryxvals'] = revealed['x']
    dat_save['queryyvals'] = revealed['y']
    filename = projectpath + "res/" + run_name
    pickle.dump(dat_save, open(filename + ".p", "wb"))


def main():
    problem = sys.argv[1]
    training_size = int(sys.argv[2])
    test_size = int(sys.argv[3])
    projectpath = sys.argv[4]
    seed = int(sys.argv[5])
    criterion = sys.argv[6]
    active_learning_steps = int(sys.argv[7])
    active_learning_func = choose_criterion(criterion)
    fit_model = choose_fit(problem)
    active_learning(problem, training_size, test_size,
                    projectpath, seed, active_learning_func, active_learning_steps, fit_model, criterion)


if __name__ == "__main__":
    main()

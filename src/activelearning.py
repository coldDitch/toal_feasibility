import sys
import pickle
import numpy as np
import random
from util import choose_fit, generate_datasets, generate_multidecision_dataset, generate_params, shadedplot, bootstrap_results
import matplotlib
import scipy
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from GPyOpt.methods import BayesianOptimization
from GPy.models import GPRegression

# matplotlib.use('tkagg')
PLOT_DATA_AND_MODEL = True
PLOT_EXPECTED_ENTROPY = False


def random_sampling(samples, fit_model, data):
    return random.randint(0, len(data['query_x'])-1)


def uncertainty_sampling_y(samples, fit_model, data):
    var = np.var(samples['py'], axis=0)
    return np.argmax(var)

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


def toal(samples, fit_model, data, objective_utility, entropy_fun):
    # toal function for finding the
    nx = len(data['query_x'])  # number of potential queries

    def f(x):
        i = np.where(np.isclose(data['query_x'], x).reshape(-1))[0][0]
        expected_entropy = 0
        # Gauss-Hermite quadrature to compute the integral
        points, weights = np.polynomial.hermite.hermgauss(
            32)  # should be atleast 32
        print("QUERY COV")
        print(data["query_x"][i])
        for ii, yy in enumerate(points):
            # predicted mean and standard deviation of point x
            mu_x, sd_x = np.mean(samples['py'][i]), np.std(samples['py'][i])
            y_star = np.sqrt(2)*sd_x*yy + mu_x  # for substitution

            # create new training set for the model
            train_x = np.append(data['train_x'], data['query_x'][i])
            train_d = np.append(data['train_d'], data['query_d'][i])
            train_y = np.append(data['train_y'], y_star)
            # fit model again
            samples_new = fit_model(data['projectpath'], train_x, train_y, train_d,
                                    data['query_x'][np.arange(nx) != i], data['query_d'][np.arange(nx) != i],
                                    data['test_x'], data['test_y'])
            H = entropy_fun(samples_new, objective_utility)
            expected_entropy += H * weights[ii] * 1/np.sqrt(np.pi)
        return expected_entropy

    if False: #TODO all decisions seperately
        # find minimum with bayesian optimization
        domain = [{'name': 'toal', 'type': 'discrete',
                   'domain': tuple(data['query_x'].ravel())}]

        initial_design_numdata = 1
        myBopt = BayesianOptimization(f=f,            # function to optimize
                                      domain=domain,
                                      initial_design_numdata=initial_design_numdata,
                                      acquisition_type='EI',
                                      exact_feval=True,
                                      normalize_Y=False,
                                      optimize_restarts=10,
                                      acquisition_weight=2,
                                      de_duplication=True)
        max_iter = nx - initial_design_numdata
        myBopt.run_optimization(max_iter=max_iter)
        myBopt.plot_acquisition()
        expected_utils = get_evaluations()
        print(expected_utils)
        # get the lowest x
        x_star = myBopt.x_opt[0]
        i_star = np.where(np.isclose(
            data['query_x'], x_star).reshape(-1))[0][0]
    else:
        # evaluate all possible query points
        expected_utils = [f(x) for x in data['query_x']]
        print(expected_utils)
        plt.plot(data['query_x'], expected_utils)
        i_star = np.argmin(expected_utils)
        x_star = data['query_x'][i_star]
        plt.title('h(' + objective_utility+')')
        plt.xlabel('queries')
        plt.ylabel('entropy of expected utility')
        plt.plot(data['query_x'], expected_utils)
        plt.show()
        plt.clf()

    print("MINIMUM")
    print(x_star)
    return i_star



def decision_ig(samples, fit_model, data):
    #TODO evaluate all decisions seperately
    return toal(samples, fit_model, data, 'decision_ig', entropy_of_maximizer_decision)
    

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


def plot_run(samples, test_x, revealed_x, revealed_d, revealed_y, run_name):
    if not PLOT_DATA_AND_MODEL:
        return
    print("Plotting")
    #plt.scatter(sam, train_y, c='b', label='observation')
    #plt.scatter(query_x, query_y, c='r', s=40, label='possible queries')
    #plt.plot(test_x, test_y, 'go', label='test set'
    decisions = int(samples['num_decisions'][0])
    np.random.seed(1234)
    for decision in range(decisions):
        res = np.empty((3, test_x.shape[0]))
        mu = samples["mu_test"][:,:,decision]
        res[0] = np.mean(mu, axis=0)
        res[1] = res[0]+np.std(mu, axis=0)
        res[2] = res[0]-np.std(mu, axis=0)
        #shadedplot(test_x, res)
        color = np.random.rand(3,)
        shadedplot(test_x, res, color=color, label='prediction d='+str(decision))
    #shadedplot(test_x, yc1_res, color='b')
    #plt.plot(test_x, np.mean(samples['mu_test'], axis=0), 'm-')
    # hack the true model plotting
    #plt.plot(test_x, slope*test_x + intercept, 'g-')
        if revealed_x.shape[0] > 0:
            rev_ind = [revealed_d==decision+1]
            plt.scatter(revealed_x[rev_ind], revealed_y[rev_ind], color=color, label='query d='+str(decision))
    plt.legend()
    plt.savefig('./plots/'+run_name+'.png')
    plt.show()
    plt.clf()


def decision_acc(samples, test_x, test_y):
    correct_count = 0
    for i in range(test_y.shape[0]):
        best_decision = 0
        model_decision = 0
        decision_util = -np.infty
        model_util = -np.infty
        for j in range(test_y.shape[1]):
            if test_y[i, j] > decision_util:
                decision_util = test_y[i, j]
                best_decision = j
            mu_util = np.mean(samples['mu_test'][:, i, j])
            if mu_util > model_util:
                model_util = mu_util
                model_decision = j
        if model_decision == best_decision:
            correct_count += 1
    return correct_count / test_y.shape[0]

def save_data(dat_save, samples, test_x, test_y):
    print("SAVING")
    dat_save["logl"].append(np.mean(np.exp(samples['logl'])))
    dat_save["acc"].append(decision_acc(samples, test_x, test_y))


def active_learning(problem, training_size, test_size, projectpath, seed, active_learning_func, steps, fit_model, criterion):
    np.random.seed(seed)
    run_name = problem + '-' + criterion + "-" + \
        str(training_size) + "-" + str(test_size) + \
        "-" + str(steps) + "-" + str(seed)
    train, test = generate_multidecision_dataset(
        problem, training_size, test_size, seed)
    (train_x, train_y, train_d), (query_x, query_y, query_d) = train
    test_x, test_y = test
    sort_index = np.argsort(test_x)
    test_y = test_y[sort_index]
    test_x = test_x[sort_index]
    sort_index = np.argsort(query_x)
    query_y = query_y[sort_index]
    query_d = query_d[sort_index]
    query_x = query_x[sort_index]

    # true probability of censoring

    print("missing shape")
    print(query_x.shape)
    print("observed shape")
    print(train_x.shape)
    dat_save = {
        "logl": [],
        "acc": [],
        "queryindices": [],
        "querydvals": [],
        "queryxvals": [],
        "queryyvals": []
    }
    samples = fit_model(projectpath, train_x, train_y, train_d, query_x, query_d, test_x,
                        test_y)
    print(samples)
    revealed_x = np.empty(0)
    revealed_d = np.empty(0)
    revealed_y = np.empty(0)
    plot_run(samples, test_x, revealed_x, revealed_d, revealed_y, run_name+'-0')
    save_data(dat_save, samples, test_x, test_y)
    for iteration in range(steps):
        data = {'projectpath': projectpath,
                'train_x': train_x,
                'train_y': train_y,
                'train_d': train_d,
                'query_x': query_x,
                'query_d': query_d,
                'test_x': test_x,
                'test_y': test_y}
        new_ind = active_learning_func(samples, fit_model, data)
        dat_save["queryindices"].append(new_ind)
        dat_save["queryxvals"].append(query_x[new_ind])
        dat_save["querydvals"].append(query_d[new_ind])
        dat_save["queryyvals"].append(query_y[new_ind])
        print("Iteration " + str(iteration) + ". Acquire point at index " +
              str(new_ind) + ": x=" + str(query_x[new_ind]))
        train_x = np.append(train_x, query_x[new_ind])
        train_y = np.append(train_y, query_y[new_ind])
        train_d = np.append(train_d, query_d[new_ind])
        revealed_x = np.append(revealed_x, query_x[new_ind])
        revealed_d = np.append(revealed_d, query_d[new_ind])
        revealed_y = np.append(revealed_y, query_y[new_ind])
        query_x = np.delete(query_x, new_ind)
        query_d = np.delete(query_d, new_ind)
        query_y = np.delete(query_y, new_ind)
        samples = fit_model(projectpath, train_x, train_y, train_d, query_x, query_d, test_x,
                            test_y)
        plot_run(samples, test_x, revealed_x, revealed_d, revealed_y, run_name+'-'+str(iteration))
        save_data(dat_save, samples, test_x, test_y)
    print(dat_save)
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

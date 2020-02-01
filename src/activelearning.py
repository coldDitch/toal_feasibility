import sys
import pickle
import numpy as np
import random
from util import choose_fit, generate_datasets, generate_params, shadedplot, bootstrap_results
import matplotlib
import scipy
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from GPyOpt.methods import BayesianOptimization
from GPy.models import GPRegression

# matplotlib.use('tkagg')
PLOT_ENTROPY_EVALS = False
PLOT_DATA_AND_MODEL = True
PLOT_EXPECTED_ENTROPY = True


def random_sampling(samples, fit_model, data):
    return random.randint(0, len(data['query_x'])-1)


def uncertainty_sampling_y(samples, fit_model, data):
    var = np.var(samples['py'], axis=0)
    return np.argmax(var)


def estimate_bandwidth(samples):
    # approximation for minimizing integrated mse
    return np.min((np.std(samples), (np.quantile(samples, 0.75)-np.quantile(samples, 0.25))/1.34)) * 0.9 * np.power(len(samples), -0.2)


def estimate_entropy_1D(sampledata, name):
    # estimates entropy of dataset where dimension 1 are samples from the distribution and
    # dimension 2 for different samples
    samples = sampledata[name]
    ntarget = samples.shape[1]
    entropy = np.zeros(ntarget)
    if PLOT_ENTROPY_EVALS:
        print(name)
        print("ESTIMATES")
        print(ntarget)
    for i in range(ntarget):
        # approximate bandwidth for minimizing the mean integrated squared error
        bandwidth = estimate_bandwidth(samples[:, i])
        # kernel density estimation and then Monte Carlo estimate of entropy
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(
            samples[:, i].reshape(-1, 1))
        y = kde.sample(500)
        entropy[i] = -1*np.mean(kde.score_samples(y))
# plot kde for debugging
        if PLOT_ENTROPY_EVALS:
            print("bandwidth")
            print(bandwidth)
            print("entropy of utility " + str(i) + ": " + str(entropy[i]))
            fig, ax = plt.subplots()
            X_plot = np.linspace(np.min(samples[:, i]), np.max(
                samples[:, i]), 1000)[:, np.newaxis]
            log_dens = kde.score_samples(X_plot)
            ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
                    label="Gaussian kernel")
            ax.legend(loc='upper left')
            ax.plot(samples[:, i], -0.005 - 0.01 *
                    np.random.random(samples.shape[0]), '+k')
            plt.show()
    return(np.mean(entropy))


def mserisk(sampledata, name):
    # integral of f(x,\theta) = p(y|\theta) (y - y*)^2 dy
    decision = np.mean(sampledata['mu_test'], axis=0)
    util = - \
        np.square(decision - sampledata['mu_test']) - \
        sampledata['sigmay'].reshape(-1, 1)
    return estimate_entropy_1D({"util": util}, "util")


def averagerisk(sampledata, name):
    # integral of f(x,\theta) = p(y|\theta) |y - y*| dy
    decision = np.mean(sampledata['mu_test'], axis=0)
    mu = sampledata['mu_test']
    var = sampledata['sigmay'].reshape(-1, 1)
    ev1 = decision - var + 2 * \
        np.exp(-np.square(decision - mu)/(2*var))*np.sqrt(2/np.pi)*np.sqrt(var)
    ev2 = (decision-mu) * \
        scipy.special.erf((decision - mu)/np.sqrt(2)/np.sqrt(var))
    ev3 = (-decision+mu) * \
        scipy.special.erfc((decision - mu)/np.sqrt(2)/np.sqrt(var))
    util = 1/2*(ev1 + ev2 + ev3)
    return estimate_entropy_1D({'util': util}, 'util')


def sixthpoly(sampledata, name):
    # integral of f(x,\theta) = p(y|\theta) (y - y*)^6 dy
    decision = np.mean(sampledata['mu_test'], axis=0)
    mu = sampledata['mu_test']
    var = sampledata['sigmay'].reshape(-1, 1)
    util = np.power(decision - mu, 6) + 15 * np.power(decision-mu, 4)*var + \
        45*np.power(decision-mu, 2)*np.power(var, 2) + 15*np.power(var, 3)
    return estimate_entropy_1D({'util': util}, 'util')


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
            train_y = np.append(data['train_y'], y_star)
            # fit model again
            samples_new = fit_model(data['projectpath'], train_x, train_y,
                                    data['query_x'][np.arange(nx) != i],
                                    data['test_x'], data['test_y'])
            H = entropy_fun(samples_new, objective_utility)
            expected_entropy += H * weights[ii] * 1/np.sqrt(np.pi)
        return expected_entropy

    if nx > 5:
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


def toal_sixth(samples, fit_model, data):
    return toal(samples, fit_model, data, 'sixthtoal', sixthpoly)


def toal_msecost(samples, fit_model, data):
    return toal(samples, fit_model, data, 'msetoal', mserisk)


def toal_avgcost(samples, fit_model, data):
    return toal(samples, fit_model, data, 'avgtoal', averagerisk)


def eig(samples, fit_model, data):
    return toal(samples, fit_model, data, 'py_test', estimate_entropy_1D)


def choose_criterion(criterion):
    # choose acquisition criterion which returns the index for the next acquisition
    if criterion == "random":
        return random_sampling
    elif criterion == "uncer_y":
        return uncertainty_sampling_y
    elif criterion == "sixthtoal":
        return toal_sixth
    elif criterion == "msetoal":
        return toal_msecost
    elif criterion == "avgtoal":
        return toal_avgcost
    elif criterion == "eig":
        return eig
    else:
        print("Activelearning not specified correctly")
        return


def plot_run(train_x, train_y, query_x, query_y, test_x, test_y, samples, revealed_x, revealed_y, run_name):
    print("Plotting")
    print(query_x.shape)
    print(revealed_x.shape)
    print(test_x.shape)
    print(train_x.shape)
    plt.scatter(train_x, train_y, c='b', label='observation')
    plt.scatter(query_x, query_y, c='r', s=40, label='possible queries')
    plt.plot(test_x, test_y, 'go', label='test set')
    res = np.empty((3, test_x.shape[0]))
    res[0] = np.mean(samples['mu_test'], axis=0)
    res[1] = res[0]+np.std(samples['mu_test'], axis=0)
    res[2] = res[0]-np.std(samples['mu_test'], axis=0)
    #shadedplot(test_x, res)
    shadedplot(test_x, res, color='r', label='predictive model')
    #shadedplot(test_x, yc1_res, color='b')
    #plt.plot(test_x, np.mean(samples['mu_test'], axis=0), 'm-')
    # hack the true model plotting
    #plt.plot(test_x, slope*test_x + intercept, 'g-')
    if revealed_x.shape[0] > 0:
        plt.plot(revealed_x, revealed_y, 'y*', label='revealed queries')
    plt.legend()
    plt.savefig('./plots/'+run_name+'.png')
    plt.show()
    plt.clf()


def save_data(dat_save, samples, test_y):
    print("SAVING")
    dat_save["logl"].append(np.mean(np.exp(samples['logl'])))
    dat_save["y_ent"].append(estimate_entropy_1D(samples, 'py_test'))
    dat_save["y_mse"].append(
        np.mean(np.square(np.mean(samples['mu_test'], axis=0) - test_y)))
    dat_save["y_avg"].append(
        np.mean(np.abs(np.mean(samples['mu_test'], axis=0) - test_y)))
    dat_save["mseutil"].append(mserisk(samples, ""))
    dat_save["sixthutil"].append(sixthpoly(samples, ""))
    dat_save["avgutil"].append(averagerisk(samples, ""))


def active_learning(problem, training_size, test_size, projectpath, seed, active_learning_func, steps, fit_model, criterion):
    np.random.seed(seed)
    run_name = problem + '-' + criterion + "-" + \
        str(training_size) + "-" + str(test_size) + \
        "-" + str(steps) + "-" + str(seed)
    train, test = generate_datasets(
        problem, training_size, test_size, seed)
    (train_x, train_y), (query_x, query_y) = train
    test_x, test_y = test
    sort_index = np.argsort(test_x)
    test_y = test_y[sort_index]
    test_x = test_x[sort_index]
    # true probability of censoring

    print("missing shape")
    print(query_x.shape)
    print("observed shape")
    print(train_x.shape)
    dat_save = {
        "logl": [],
        "logl1": [],
        "y_ent": [],
        "mseutil": [],
        "avgutil": [],
        "sixthutil": [],
        "y_mse": [],
        "y_avg": [],
        "y_sixth": [],
        "queryindices": [],
        "queryxvals": [],
        "queryyvals": []
    }
    samples = fit_model(projectpath, train_x, train_y, query_x, test_x,
                        test_y)
    revealed_x = np.empty(0)
    revealed_y = np.empty(0)
    plot_run(train_x, train_y, query_x, query_y, test_x, test_y,
             samples, revealed_x, revealed_y, run_name+'-0')
    save_data(dat_save, samples, test_y)
    for iteration in range(steps):
        data = {'projectpath': projectpath, 'train_x': train_x, 'train_y': train_y,
                'query_x': query_x,
                'test_x': test_x,
                'test_y': test_y}
        new_ind = active_learning_func(samples, fit_model, data)
        dat_save["queryindices"].append(new_ind)
        dat_save["queryxvals"].append(query_x[new_ind])
        dat_save["queryyvals"].append(query_y[new_ind])
        print("Iteration " + str(iteration) + ". Acquire point at index " +
              str(new_ind) + ": x=" + str(query_x[new_ind]))
        train_x = np.append(train_x, query_x[new_ind])
        train_y = np.append(train_y, query_y[new_ind])
        revealed_x = np.append(revealed_x, query_x[new_ind])
        revealed_y = np.append(revealed_y, query_y[new_ind])
        query_x = np.delete(query_x, new_ind)
        query_y = np.delete(query_y, new_ind)
        samples = fit_model(projectpath, train_x, train_y, query_x, test_x,
                            test_y)
        plot_run(train_x, train_y, query_x, query_y, test_x, test_y,
                 samples, revealed_x, revealed_y, run_name + '-' + str(iteration + 1))
        save_data(dat_save, samples, test_y)
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

import numpy as np
import matplotlib.pyplot as plt
import config
from sklearn.neighbors import KernelDensity

def plot_all(samples, test, train, run_name):
    """
        Plots current model,
        utility density and probability mass function of D_best at target index
    """
    if config.save_plots or config.show_plots:
        target = 0
        plot_argmax(samples, target, run_name)
        plot_test_point(samples, target, run_name)
        plot_run(samples, test, train, run_name)


def plot_run(samples, test, train, run_name):
    """
    Plots the outcomes and training data as function of each covariate
    """
    test['x'] = test['x'].reshape(test['x'].shape[0], -1)
    train['x'] = train['x'].reshape(train['x'].shape[0], -1)
    for cov in range(test['x'].shape[1]):
        marg_dat = {
            'x': train['x'][:, cov],
            'y': train['y'],
            'd': train['d']
        }
        sort_by_covariates(marg_dat)
        decisions = config.decision_n
        np.random.seed(1234)
        for decision in range(decisions):
            res = np.empty((3, test['x'].shape[0]))
            mu = samples["u_bar"][:, :, decision]
            plot_dat = {
                'x': test['x'][:, cov],
                'mu': mu.T
            }
            sort_by_covariates(plot_dat)
            res[0] = np.mean(plot_dat['mu'].T, axis=0)
            res[1] = res[0]+np.std(plot_dat['mu'].T, axis=0)
            res[2] = res[0]-np.std(plot_dat['mu'].T, axis=0)
            color = np.random.rand(3,)
            shadedplot(plot_dat['x'], res, color=color, label='d='+str(decision))
            plt.scatter(marg_dat['x'][decision+1 == marg_dat['d']], marg_dat['y'][decision+1 == marg_dat['d']], color=color)
        plt.xlabel('covariate x', fontsize=20)
        plt.ylabel('utility u', fontsize=20)
        plt.rc('font', size=20)
        plt.rc('axes', labelsize=20)
        plt.rc('legend', fontsize=20)
        #plt.axvline(0.4, -10, 10, label='target x')
        plt.legend(loc='lower left')
        if config.show_plots:
            plt.show()
        if config.save_plots:
            plt.savefig('./plots/cov'+str(cov)+'-'+run_name+'.png')
        plt.clf()

def plot_test_point(samples_data, test_point, run_name):
    """
    Plots utility densities at a current testpoint
    """
    np.random.seed(1234)
    plt.xlabel('utility u', fontsize=20)
    plt.ylabel('p(u)', fontsize=20)
    samples = samples_data['u_bar']
    for d in range(config.decision_n):
        color = np.random.rand(3,)
        bandwidth = estimate_bandwidth(samples[:, test_point, d])
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(
            samples[:, test_point, d].reshape(-1, 1))
        X_plot = np.linspace(np.min(samples[:, test_point, d]),
            np.max(samples[:, test_point, d]), 500)[:, np.newaxis]
        log_dens = kde.score_samples(X_plot)
        plt.plot(X_plot[:, 0], np.exp(log_dens), '-', label='d='+str(d), color=color)
        plt.fill(X_plot[:, 0], np.exp(log_dens), alpha=0.3, color=color)
    if config.show_plots:
        plt.show()
    if config.save_plots:
        plt.savefig('./plots/testpoint'+str(test_point)+'-'+run_name+'.png')
    plt.clf()


def plot_argmax(sampledata, target, run_name):
    """
    Plots probability mass function of best decisions (D_best) for target covariate
    """
    np.random.seed(1234)
    decisions = config.decision_n
    samples = sampledata["u_bar"]
    probs = []
    colors = []
    labels = []
    for decision in range(decisions):
        maximizers = np.ones(samples.shape[0], dtype=bool)
        for d in range(decisions):
            maximizers = np.logical_and((samples[:, target, d] <= samples[:, target, decision]), maximizers)
        prob = np.sum(maximizers) / len(maximizers)
        color = np.random.rand(3,)
        probs.append(prob)
        colors.append(color)
        labels.append('d='+str(decision))
    print(labels)
    print(probs)
    plt.bar(np.array(labels), np.array(probs), color=colors)
    plt.ylabel('Probability of maximizing utility', fontsize=20)
    if config.show_plots:
        plt.show()
    if config.save_plots:
        plt.savefig('./plots/probmass'+run_name+'.png')
    plt.clf()


def shadedplot(x, y, fill=True, label='', color='b'):
    """
    Forms a plot with confidence boundaries

    y[0,:] mean, median etc; in the middle
    y[1,:] lower
    y[2,:] upper
    """
    p = plt.plot(x, y[0, :], label=label, color=color)
    c = p[-1].get_color()
    if fill:
        plt.fill_between(x, y[1, :], y[2, :], color=c, alpha=0.25)


def sort_by_covariates(dat):
    """
    sort data based on covariates for plotting
    """
    sort_index = np.argsort(dat['x'])
    for d in dat.keys():
        dat[d] = dat[d][sort_index]


def mean_conf(dat):
    """
    mean and it's standard deviation
    """
    N, Nq = dat.shape
    ret = np.empty((3, Nq))
    ret[0, :] = np.quantile(dat, 0.5, axis=0)
    ret[1, :] = np.quantile(dat, 0.25, axis=0)
    ret[2, :] = np.quantile(dat, 0.75, axis=0)
    return ret


def estimate_bandwidth(samples):
    """
    approximation for minimizing integrated mse for density
    """
    return np.min((np.std(samples), (np.quantile(samples, 0.75)-np.quantile(samples, 0.25))/1.34)) * 0.9 * np.power(len(samples), -0.2)


def plot_density_estimate(samples, kde, i, d):
    """
    plots the utility density for covariate at index i and decision d
    """
    if config.save_plots or config.show_plots:
        X_plot = np.linspace(np.min(samples[:, i, d]), np.max(
            samples[:, i, d]), 500)[:, np.newaxis]
        log_dens = kde.score_samples(X_plot)
        plt.plot(X_plot[:, 0], np.exp(log_dens), '-',
                label="Gaussian kernel")
        plt.plot(samples[:, i, d], -0.005 - 0.01 *
                np.random.random(samples.shape[0]), '+k', label='stan samples')
        plt.xlabel('utility')
        plt.ylabel('p(u)')
        plt.title('cov: '+str(i)+' decision: '+str(d))
        plt.legend(loc='upper left')
        if config.show_plots:
            plt.show()
        else:
            plt.savefig('./plots/eig_density'+str(i)+("_")+str(d))
        plt.clf()

def plot_expected_values(data, expected_utils, objective_utility):
    """
    plots expected utilities as function of covariates
    """
    cov = 0 # plot only along a single covariate
    if config.save_plots or config.show_plots:
        for d in range(1, config.decision_n+1):
            color = np.random.rand(3,)
            plt.scatter(data["query"]["x"][:, cov][data["query"]["d"]==d], np.array(expected_utils)[data["query"]["d"]==d],color=color, label="d="+str(d-1))
        plt.legend()
        plt.title("h(" + objective_utility+")")
        plt.xlabel("query points")
        plt.ylabel("expected entropy of decisions")
        if config.show_plots:
            plt.show()
        else:
            plt.savefig("./plots/entropies.png")
        plt.clf()

def plot_query_fun(y_stars, entropies):
    """
    plots entropy as function of potential label y_star
    """
    plt.plot(y_stars, entropies)
    plt.xlabel("y_star")
    plt.ylabel("H(D_best)")
    plt.savefig("./plots/expectation.png")
    plt.show()

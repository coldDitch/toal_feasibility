"""
This utilities file contains preprocessing functions for ACICC2016 dataset

"""
import math
import numpy as np
import numpy.random as rndm
import pandas as pd
import config


def generate_dataset(seed):
    """
    Choose either synthetically generated function on ACICC dataset
    """
    if config.dataset == 'acic':
        return generate_acic_dataset(config.train_n, config.test_n, config.query_n, config.decision_n, seed)
    elif config.dataset == 'synthetic':
        return generate_multidecision_dataset(config.train_n, config.test_n, config.query_n, config.decision_n, seed)
    else:
        print("CHOOSE PROPER DATASET")
        return None


def covariate_matrix():
    """
    Read covariate file and transform it to matrix (n, d)
    where n is number of observations and d number of features
    """
    df = pd.read_csv(config.acic_path + 'x.csv')
    col = []
    for key in df.keys():
        covariate = df[key].values
        if is_categorical(covariate):
            col.append(to_binary(covariate))
        else:
            col.append(covariate.reshape(-1,1))
    col = np.concatenate(col, axis=1)
    return col


def acic_covariates():
    """
    Read and normalize features to make GP lengthscales comparable
    """
    mat = covariate_matrix()
    mat, _, _ = normalize(mat)
    return mat


def acic_labels(fil):
    """
    Read file (fil) and return vector of real outcomes (n), treatments chosen
    for those outcomes (n) and potential outcomes (n, 2) which has the outcome for boft treatment
    decisions in the file. To test with less noise y0 and y1 can be replaced by expected values mu0, mu1.
    """
    df = pd.read_csv(config.acic_path + fil)
    potential_outcomes = df[['y0', 'y1']].values
    treatments = df['z'].values
    outcomes = np.array([potential_outcomes[i, treatments[i]] for i in range(len(treatments))])
    treatments = treatments + 1
    return outcomes, treatments, potential_outcomes


def generate_acic_dataset(training_size, test_size, query_size, decision_n, seed):
    """
    Combine files to form multidecision acic dataset, sample training, test
    and query datasets from file without replacement.
    """
    rndm.seed(seed)
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
    indexes = rndm.choice(len(treatments), training_size + test_size + num_queries, replace=False)
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
    revealed = revealed_data()
    return train, query, test, revealed


def is_categorical(val):
    """
    Check if ACICC feature is categorical
    """
    return val.dtype == np.dtype('O')


def to_binary(val):
    """
    Transforms categorical variable (n), into binary matrix (n, c)
    where c is number of categories
    """
    uniques = np.unique(val)
    mat = np.zeros((len(val), len(uniques)))
    for i in range(len(uniques)):
        byte_arr = (val == uniques[i]).astype('b')
        mat[:, i] = byte_arr
    return mat

def normalize(mat):
    """
    Force feature mean to 0 and standard deviation < 1
    """
    mean = np.mean(mat, axis=0)
    l = len(mean) if type(mean) is np.ndarray else 1
    mat = mat - mean
    std = np.array(np.std(mat, axis=0))
    std = np.max((std, np.ones(l)))
    mat = mat / std
    return mat, mean, std

def generate_params(seed, dimensions):
    """
    Generate parameters for synthetic data, coef_1 is slope and coef_2 is intercept
    """
    rndm.seed(seed)
    coef_1 = rndm.uniform(-20, 20, size=(dimensions, 1))
    coef_2 = rndm.uniform(-1, 1)
    return coef_1, coef_2

def revealed_data():
    """
    Empty set for revealed data
    """
    revealed = {
        'x': np.empty(0),
        'd': np.empty(0),
        'y': np.empty(0)
    }
    return revealed


def generate_multidecision_dataset(training_size, test_size, query_size, decision_n, seed):
    """
    Generate synthetic dataset
    """
    num_queries = query_size
    std = config.std
    dimensions = config.synthetic_dim
    noisy = config.noisy
    # query set from which model chooses x and d, for which we reveal y
    query_x = covariate_dist(num_queries, dimensions)
    query_d = rndm.randint(1, decision_n+1, num_queries)
    query_y = np.zeros(num_queries)
    # initial data for training
    train_x = covariate_dist(training_size, dimensions)
    train_d = rndm.randint(1, decision_n+1, training_size)
    train_y = np.zeros(training_size)
    # test set, for test set outcome for all decisions are known to find the best decision
    test_x = covariate_dist(test_size, dimensions)
    test_y = np.zeros((test_size, decision_n))
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

    if config.model=='quadratic':
        print('quadratic gen')
        quadratic_gen(noisy, query, train, test, decision_n, seed, std)
    else:
        linear_gen(noisy, query, train, test, decision_n, seed, std)




    train['y'], mean, std = normalize(train['y'])
    test['y'] = (test['y'] - mean) / std
    query['y'] = (query['y'] - mean) / std
    revealed = revealed_data()
    return train, query, test, revealed

def linear_gen(noisy, query, train, test, decision_n, seed, std):
    """
    linear dataset W . X + b = y
    """
    if noisy:
        # Only the first covariate is correlated with the outcomes
        for i in range(1, decision_n + 1):
            slope, intercept = generate_params(seed+i, 1)
            query['y'][query['d'] == i] = np.dot(query['x'][query['d'] == i, 0].reshape(-1, 1), slope).ravel()
            query['y'][query['d'] == i] += intercept + rndm.normal(0, std, len(query['x'][query['d'] == i, 0]))
            train['y'][train['d'] == i] = np.dot(train['x'][train['d'] == i, 0].reshape(-1, 1), slope).ravel()
            train['y'][train['d'] == i] += intercept + rndm.normal(0, std, len(train['x'][train['d'] == i, 0]))
            test['y'][:,i-1] = test['x'][:, 0] * slope + intercept
    else:
        # All the covaries are dependent on the outcome
        for i in range(1, decision_n + 1):
            slope, intercept = generate_params(seed+i, dimensions)
            query['y'][query['d'] == i] = np.dot(query['x'][query['d'] == i, :], slope).ravel()
            query['y'][query['d'] == i] += intercept + rndm.normal(0, std, len(query['x'][query['d'] == i, :]))
            train['y'][train['d'] == i] = np.dot(train['x'][train['d'] == i, :], slope).ravel()
            train['y'][train['d'] == i] += intercept + rndm.normal(0, std, len(train['x'][train['d'] == i, :]))
            test['y'][:, i-1] = np.dot(test['x'], slope).ravel() + intercept

def quadratic_gen(noisy, query, train, test, decision_n, seed, std):
    """
    quadratic dataset W_1 . X^2 + W_2 . X + b = y
    """
    if noisy:
        # Only the first covariate is correlated with the outcomes
        for i in range(1, decision_n + 1):
            quad_a, _ = generate_params(seed+i, 1)
            slope, intercept = generate_params(seed+i, 1)
            query['y'][query['d'] == i] = np.dot(np.power(query['x'][query['d'] == i, 0].reshape(-1, 1), 2), quad_a).ravel()
            query['y'][query['d'] == i] += np.dot(query['x'][query['d'] == i, 0].reshape(-1, 1), slope).ravel()
            query['y'][query['d'] == i] += intercept + rndm.normal(0, std, len(query['x'][query['d'] == i, 0]))
            train['y'][train['d'] == i] = np.dot(np.power(train['x'][train['d'] == i, 0].reshape(-1, 1), 2), quad_a).ravel()
            train['y'][train['d'] == i] += np.dot(train['x'][train['d'] == i, 0].reshape(-1, 1), slope).ravel()
            train['y'][train['d'] == i] += intercept + rndm.normal(0, std, len(train['x'][train['d'] == i, 0]))
            test['y'][:,i-1] = np.power(test['x'][:, 0], 2) * quad_a + test['x'][:, 0] * slope + intercept
    else:
        # All the covaries are dependent on the outcome
        for i in range(1, decision_n + 1):
            quad_a, _ = generate_params(seed+i, dimensions)
            slope, intercept = generate_params(seed+i, dimensions)
            query['y'][query['d'] == i] = np.dot(np.power(query['x'][query['d'] == i, :].reshape(-1, 1), 2), quad_a).ravel()
            query['y'][query['d'] == i] += np.dot(query['x'][query['d'] == i, :], slope).ravel()
            query['y'][query['d'] == i] += intercept + rndm.normal(0, std, len(query['x'][query['d'] == i, :]))
            train['y'][train['d'] == i] = np.dot(np.power(train['x'][train['d'] == i, :].reshape(-1, 1), 2), quad_a).ravel()
            train['y'][train['d'] == i] += np.dot(train['x'][train['d'] == i, :], slope).ravel()
            train['y'][train['d'] == i] += intercept + rndm.normal(0, std, len(train['x'][train['d'] == i, :]))
            test['y'][:, i-1] = np.dot(np.power(test['x'], 2), quad_a) + np.dot(test['x'], slope).ravel() + intercept

def covariate_dist(num_covariates, num_features):
    """
    Synthetic feature matrix with uniformly random data on [-0.5, 0.5]^d
    """
    return rndm.random((num_covariates, num_features)) - 0.5

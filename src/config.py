"""
Configuration file for running the script
"""
# synthetic, acic
dataset = 'synthetic'

# stanmodel name, linear, se_gp, ard_se_gp
model = 'linear'

# number of potential decisions (treatments)
decision_n = 4

# number of training, test and potential querypoints, has to be lower than 4802 for ACIC
train_n = 20
test_n = 10
query_n = 5

# Save plots from runA
show_plots = True

# Save plots from run
save_plots = True

# Runs full diagnostics for stanmodel, this is 
run_diagnostics = False

### ACICC data parameters
acic_path = '../../datasets/data_cf_all/'
# list of files which is used to get enough decisions
acic_files = ['10/zymu_236.csv', '10/zymu_7692299.csv', '10/zymu_7692308.csv', '10/zymu_7692384.csv']

###  Synthetic data parameters
# number of features in synthetic data
synthetic_dim = 2

# If noisy this means that only the first covariate has effect on the outcome,
# if not all covariates are correlated with the outcome
noisy = False

# Standard deviation added to synthetic data, data is normalized afterwards so 
# this essentially determines the ration of effect and noise
std = 1

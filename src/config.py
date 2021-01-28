"""
Configuration file for running the script
"""
# synthetic, acic
dataset = 'synthetic'

# stanmodel name, linear, quadratic, se_gp, ard_se_gp
model = 'quadratic'

# number of potential decisions (treatments)
decision_n = 4

# number of training, test and potential querypoints, has to be lower than 4802 for ACIC
train_n = 12
test_n = 100
query_n = 20

# Save plots from run
show_plots = True

# Save plots from run
save_plots = False

# Runs full diagnostics for stanmodel
run_diagnostics = True

### ACICC data parameters
acic_path = '../../datasets/data_cf_all/'
# list of files which is used to get enough decisions
acic_files = ['10/zymu_236.csv', '10/zymu_7692299.csv', '10/zymu_7692308.csv', '10/zymu_7692384.csv']

###  Synthetic data parameters
# number of features in synthetic data
synthetic_dim = 1

# If noisy this means that only the first covariate has effect on the outcome,
# if not all covariates are correlated with the outcome
noisy = True

# Standard deviation added to synthetic data, data is normalized afterwards so
# this essentially determines the ration of effect and noise
std = 1

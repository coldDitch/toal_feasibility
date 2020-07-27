

data {
    int<lower=0> n;                     // number of data points
    int<lower=1> nd;   // number of decisions
    int<lower=1> k;
    matrix[n, k] x;                       // explanatory variable
    vector[n] y;          // response variable
    vector[n] d;
    int<lower=0> cn;
    matrix[cn, k] cx;
    vector[cn] cd;
    int<lower=0> ntest;
    matrix[ntest, k] xtest;
    matrix[ntest, nd] ytest;
}
parameters {
    matrix[nd, k] beta;
    vector[nd] alpha;
    vector<lower=1e-3>[nd] sigmay;
}
model {
    sigmay ~ inv_gamma(2, 2);
    to_vector(beta) ~ normal(0, 1);
    alpha ~ normal(0, 1);
    for(j in 1:nd){
        for(i in 1:n) {
            if(j == d[i]){
                y[i] ~ normal(beta[j] * x[i]' + alpha[j], sigmay[j]);
            }
        }
    }
}
generated quantities {
    real logl[ntest];
    matrix[ntest, nd] mu_test;
    real py[cn];
    real mu[cn];
    for(i in 1:ntest){
        for(j in 1:nd){
            mu_test[i, j] = beta[j] * xtest[i]' + alpha[j];
            logl[i] = normal_lpdf(ytest[i, j]|mu_test[i, j], sigmay[j]);
        }
    }
    if(cn > 0) {
        for(i in 1:cn) {
            for(j in 1:nd) {
                if(j == cd[i]){
                    mu[i] = beta[j] * cx[i]' + alpha[j];
                    py[i] = normal_rng(mu[i], sigmay[j]);
                }
            }
        }
    }
}

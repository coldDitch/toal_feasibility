data {
    int<lower=0> n;                     // number of data points
    int<lower=1> k;
    int<lower=1> nd;   // number of decisions
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
    real sigmay;
}

model {
    real eta;
    sigmay ~ inv_chi_square(0.1);
    beta[1] ~ normal(0, 10);
    alpha[1] ~ normal(0, 10);
    for (i in 2:nd) {
      beta[i] ~ normal(0, 1);
      alpha[i] ~ normal(0, 1);
    }
    for(j in 1:nd){
        for(i in 1:n) {
            if(j == d[i]){
              eta = x[i] * beta[1]' + alpha[1];
              if(j != 1){
                eta = eta + x[i] * beta[j]' + alpha[j];
              }
              y[i] ~ normal(eta, sigmay);
            }
        }
    }
}

generated quantities {
    real logl[ntest];
    matrix[ntest, nd] u_bar;
    real py[cn];
    real mu[cn];
    for(i in 1:ntest){
        for(j in 1:nd){
            u_bar[i, j] = xtest[i] * beta[j]' + alpha[j];
            if (j != 1) {
              u_bar[i, j] = u_bar[i, j] + xtest[i] * beta[1]' + alpha[1];
            }
            logl[i] = normal_lpdf(ytest[i, j]|u_bar[i, j], sigmay);
        }
    }
    if(cn > 0) {
        for(i in 1:cn) {
            for(j in 1:nd) {
                if(j == cd[i]){
                    mu[i] = cx[i] * beta[j]' + alpha[j];
                    if (j != 1) {
                      mu[i] = mu[i] + cx[i] * beta[1]' + alpha[1];
                    }
                    py[i] = normal_rng(mu[i], sigmay);
                }
            }
        }
    }
}

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
    vector<lower=1e-3>[nd] sigmay;
}

model {
    real eta;
    sigmay ~ normal(1, 10);
    to_vector(beta) ~ normal(0, 10);
    alpha ~ normal(0, 10);
    for(j in 1:nd){
        for(i in 1:n) {
            if(j == d[i]){
		eta = x[i] * beta[1]' + alpha[1];
		if(j != 1){
		  eta = eta + x[i] * beta[j]' + alpha[j];
		}
                y[i] ~ normal(eta, sigmay[j]);
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
            logl[i] = normal_lpdf(ytest[i, j]|u_bar[i, j], sigmay[j]);
        }
    }
    if(cn > 0) {
        for(i in 1:cn) {
            for(j in 1:nd) {
                if(j == cd[i]){
                    mu[i] = cx[i] * beta[j]' + alpha[j];
                    py[i] = normal_rng(mu[i], sigmay[j]);
                }
            }
        }
    }
}

functions {
    matrix psi(data int n, data vector x) {
        matrix[n,2] lpsi; 
        vector[2] xr = [-2,2]';
        for (i in 1:n) {
            for (j in 1:2) {
                lpsi[i][j] = exp(-(x[i] - xr[j])^2);
            }
        }
        return lpsi;
    }
}

data {
    int<lower=0> n;
    vector[n] x;
    vector[n] y;
    int<lower=0> cn;
    vector[cn] cx;
    int<lower=0> ntest;
    vector[ntest] xtest;
    vector[ntest] ytest;
}
parameters {
    vector[2] beta;
    real<lower=0> sigmay;
}
transformed parameters {
    vector[n] eta;
    eta = psi(n, x) * beta;
}
model {
    //priors
    beta ~ normal(0, 10);
    sigmay ~ inv_gamma(2, 2);
    //likelihood
    y ~ normal(eta, sigmay);
}
generated quantities {
    vector[ntest] logl;
    vector[cn] py;
    vector[ntest] py_test;
    vector[ntest] mu_test = psi(ntest, xtest) * beta;
    vector[cn] cens_mu = psi(cn, cx) * beta;
    for(i in 1:ntest) {
	logl[i] = normal_lpdf(ytest[i]|mu_test[i], sigmay);
        py_test[i] = normal_rng(mu_test[i], sigmay);
    }
    for(i in 1:cn) {
        py[i] = normal_rng(cens_mu[i], sigmay);
    }
}

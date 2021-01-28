functions {
    row_vector vec_pow(row_vector x, int p) {
      int n = cols(x);
      row_vector[n] x_pow;
      for (i in 1:n) {
        x_pow[i] = pow(x[i], p);
      }
    return x_pow;
    }
}

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
    matrix[nd, k] beta[2];
    vector[nd] alpha;
    real sigmay;
}

model {
    real eta;
    sigmay ~ inv_chi_square(0.1);
    beta[2][1] ~ normal(0, 10);
    beta[1][1] ~ normal(0, 10);
    alpha[1] ~ normal(0, 10);
    for (i in 2:nd) {
      beta[2][i] ~ normal(0, 2);
      beta[1][i] ~ normal(0, 2);
      alpha[i] ~ normal(0, 2);
    }
    for(j in 1:nd){
        for(i in 1:n) {
            if(j == d[i]){
              eta = vec_pow(x[i] ,2) * beta[1][1]' + x[i] * beta[2][1]' + alpha[1];
              if(j != 1){
                eta = eta + vec_pow(x[i], 2) * beta[1][j]' + x[i] * beta[2][j]' + alpha[j];
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
            u_bar[i, j] = vec_pow(xtest[i],2) * beta[1][j]' + xtest[i] * beta[2][j]' + alpha[j];

            if(j != 1){
              u_bar[i, j] = u_bar[i, j] + vec_pow(xtest[i],2) * beta[1][1]' + xtest[i] * beta[2][1]' + alpha[1];
            }
            logl[i] = normal_lpdf(ytest[i, j]|u_bar[i, j], sigmay);
        }
    }
    if(cn > 0) {
        for(i in 1:cn) {
            for(j in 1:nd) {
                if(j == cd[i]){
                    mu[i] = vec_pow(cx[i],2) * beta[1][j]' + cx[i] * beta[2][j]' + alpha[j];

                    if(j != 1){
                      mu[i] = mu[i] + vec_pow(cx[i],2) * beta[1][1]' + cx[i] * beta[2][1]' + alpha[1];
                    }
                    py[i] = normal_rng(mu[i], sigmay);
                }
            }
        }
    }
}

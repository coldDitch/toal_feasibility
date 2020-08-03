
data {
    int<lower=0> n;                     // number of data points
    int<lower=1> k;
    matrix[n, k] x;                       // explanatory variable
    vector[n] y;          // response variable
    int<lower=0> cn;
    matrix[cn, k] cx;
    int<lower=0> ntest;
    matrix[ntest, k] xtest;
    vector[ntest] ytest;
}
parameters {
    vector[k] beta;
    real alpha;
    real<lower=1e-3> sigmay;
}
model {
    sigmay ~ normal(0, 1);
    to_vector(beta) ~ normal(0, 10);
    alpha ~ normal(0, 10);
    y ~ normal(x * beta + alpha, sigmay);
}
generated quantities {
    real logl;
    vector[ntest] u_bar;
    vector[cn] mu;
    real py[cn];
    u_bar = xtest * beta + alpha;
    logl = normal_lpdf(ytest|u_bar, sigmay);
    if(cn > 0) {
        mu = cx * beta + alpha;
        py = normal_rng(mu, sigmay);
    }
}

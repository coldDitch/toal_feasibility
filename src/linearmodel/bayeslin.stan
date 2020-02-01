data {
    int<lower=0> n;                     // number of data points
    vector[n] x;                       // explanatory variable
    vector[n] y;          // response variable
    int<lower=0> cn;
    vector[cn] cx;
    int<lower=0> ntest;
    vector[ntest] xtest;
    vector[ntest] ytest;
}
parameters {
    real beta;
    real alpha;
    real<lower=1e-3> sigmay;
}
model {
    sigmay ~ normal(1, 10);
    beta ~ normal(0, 10);
    alpha ~ normal(0, 10);
    y ~ normal(beta * x + alpha, sigmay);
}
generated quantities {
    real logl[ntest];
    real py[cn];
    for (i in 1:ntest)
        logl[i] = normal_lpdf(ytest[i]|beta * xtest[i] + alpha, sigmay);
    if (cn > 0)
        py = normal_rng(beta * cx + alpha, sigmay);
}

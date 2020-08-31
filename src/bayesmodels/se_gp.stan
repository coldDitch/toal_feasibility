functions {
  vector gp_combine_rng(row_vector[] t,
                     vector y1,
                     row_vector[] x1,
                     real alpha,
                     real rho,
                     real sigma,
                     real delta) {
    int n = rows(y1);
    int T = size(t);
    vector[T] res;
    {
      vector[T] f2_mu;
      matrix[T, T] cov_f2;
      matrix[n, n] L_K;
      vector[n] K_div_y1;
      matrix[n, T] k_x1_t;
      matrix[n, T] v_pred;
      matrix[T, T] diag_delta;
      matrix[n, n] K;
      vector[T] variance_gp;
      K = cov_exp_quad(x1, alpha, rho);
      for (i in 1:n)
        K[i, i] = K[i,i] + square(sigma);
      L_K = cholesky_decompose(K);
      K_div_y1 = mdivide_left_tri_low(L_K, y1);
      K_div_y1 = mdivide_right_tri_low(K_div_y1', L_K)';
      k_x1_t = cov_exp_quad(x1, t, alpha, rho);
      f2_mu = (k_x1_t' * K_div_y1);
      v_pred = mdivide_left_tri_low(L_K, k_x1_t);
      cov_f2 = cov_exp_quad(t, alpha, rho) - v_pred' * v_pred;
      for(i in 1:T)
        cov_f2[i,i] = cov_f2[i,i] + delta;
      res = multi_normal_rng(f2_mu, cov_f2);
    }
    return res;
  }

}

data {
    int<lower=0> n;       // number of data points
    int<lower=1> k;
    row_vector[k] x[n];       // explanatory variable
    vector[n] y;          // response variable
    int<lower=0> cn;
    row_vector[k] cx[cn];
    int<lower=0> ntest;
    row_vector[k] xtest[ntest];
    vector[ntest] ytest;
}

transformed data {
  vector[n] mean_f = rep_vector(0, n);
  real delta = 1e-9;
}

parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=delta> sigma;
}

model {
  matrix[n, n] L_K;
  matrix[n, n] K;
  // priors
  rho ~ inv_gamma(2, 5);
  alpha ~ normal(0, 1);
  sigma ~ inv_gamma(2, 1);

  // likelihood computation
  K = cov_exp_quad(x, alpha, rho);
  // diagonal elements
  for (j in 1:n)
    K[j, j] = K[j, j] + sigma;
  L_K = cholesky_decompose(K);
  y ~ multi_normal_cholesky(mean_f, L_K);
}

generated quantities {
  real logl;
  vector[ntest] u_bar;
  vector[cn] mu;
  real py[cn];

  u_bar = gp_combine_rng(xtest, y, x, alpha, rho, sigma, delta);
  // not sure if correct
  logl = normal_lpdf(ytest|u_bar,sigma);
  if (cn > 0) {
    mu = gp_combine_rng(cx, y, x, alpha, rho, sigma, delta);
    py = normal_rng(mu, sigma);
  }
}

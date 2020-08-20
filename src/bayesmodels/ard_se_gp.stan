functions {

  matrix cov_exp_quad_multidim(real sq_alpha, vector rho, matrix x, int n) {
    matrix[n, n] K;
    for (i in 1:n) {
      for (j in i:n) {
        K[i, j] = sq_alpha * exp(-0.5 * dot_self((x[i] - x[j])' ./ rho));
        K[j, i] = K[i, j];
      }
    }
    return K;
  }

  matrix cov_exp_quad_two_multidim(real sq_alpha, vector rho, matrix x1, matrix x2, int n1, int n2) {
    matrix[n1, n2] K;
    for (i in 1:n1) {
      for (j in 1:n2) {
        K[i, j] = sq_alpha * exp(-0.5 * dot_self((x1[i] - x2[j])' ./ rho));
      }
    }
    return K;
  }

  vector gp_combine_rng(matrix t,
                     vector y1,
                     matrix x1,
                     real sq_alpha,
                     vector rho,
                     real sigma,
                     real delta) {
    int n = rows(y1);
    int T = rows(t);
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
      K = cov_exp_quad_multidim(sq_alpha, rho, x1, n);
      for (i in 1:n)
        K[i, i] = K[i,i] + square(sigma);
      L_K = cholesky_decompose(K);
      K_div_y1 = mdivide_left_tri_low(L_K, y1);
      K_div_y1 = mdivide_right_tri_low(K_div_y1', L_K)';
      k_x1_t = cov_exp_quad_two_multidim(sq_alpha, rho, x1, t, n, T);
      f2_mu = (k_x1_t' * K_div_y1);
      v_pred = mdivide_left_tri_low(L_K, k_x1_t);
      cov_f2 = cov_exp_quad_multidim(sq_alpha, rho, t, T) - v_pred' * v_pred;
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
    matrix[n, k] x;       // explanatory variable
    vector[n] y;          // response variable
    int<lower=0> cn;
    matrix[cn, k] cx;
    int<lower=0> ntest;
    matrix[ntest, k] xtest;
    vector[ntest] ytest;
}

transformed data {
  vector[n] mean_f = rep_vector(0, n);
  real delta = 1e-9;
}

parameters {
  vector<lower=0>[k] rho;
  real<lower=0> sq_alpha;
  real<lower=delta> sigma;
}

model {
  matrix[n, n] L_K;
  matrix[n, n] K;
  // priors
  rho ~ inv_gamma(0.8, 20);
  sq_alpha ~ normal(0, 10000);
  sigma ~ inv_gamma(2, 1);

  // likelihood computation
  K = cov_exp_quad_multidim(sq_alpha, rho, x, n);
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

  u_bar = gp_combine_rng(xtest, y, x, sq_alpha, rho, sigma, delta);
  // not sure if correct
  logl = normal_lpdf(ytest|u_bar,sigma);
  if (cn > 0) {
    mu = gp_combine_rng(cx, y, x, sq_alpha, rho, sigma, delta);
    py = normal_rng(mu, sigma);
  }
}

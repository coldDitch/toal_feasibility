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

  int count_decisions(vector d_vec, int d, int n) {
    int count = 0;
    for (i in 1:n) {
      if (d_vec[i]==d) {
        count += 1;
      }
    }
    return count;
  }

  matrix sub_covariate_matrix(matrix x, vector d_vec, int d, int n, int k) {
    int sub_dec = count_decisions(d_vec, d, n);
    matrix[sub_dec, k] x_sub;
    int ind = 1;
    for (i in 1:n) {
      if (d_vec[i] == d) {
        x_sub[ind] = x[i];
        ind += 1;
      }
    }
    return x_sub;
  }

  vector sub_outcome_vector(vector y, vector d_vec, int d, int n) {
    int sub_dec = count_decisions(d_vec, d, n);
    vector[sub_dec] y_sub;
    int ind = 1;
    for (i in 1:n) {
      if (d_vec[i] == d) {
        y_sub[ind] = y[i];
        ind += 1;
      }
    }
    return y_sub;
  }
}

data {
    int<lower=0> n;       // number of data points
    int<lower=1> nd;      // number of decisions
    int<lower=1> k;
    matrix[n, k] x;       // explanatory variable
    vector[n] y;          // response variable
    vector[n] d;
    int<lower=0> cn;
    matrix[cn, k] cx;
    int cd[cn];
    int<lower=0> ntest;
    matrix[ntest, k] xtest;
    matrix[ntest, nd] ytest;
}

transformed data {
  real delta = 1e-9;
}

parameters {
  vector<lower=0>[k] rho[nd];
  real<lower=0> sq_alpha[nd];
  real<lower=1e-3> sigma[nd];
}

model {
  // priors
  for (i in 1:nd)
    rho[i] ~ inv_gamma(1, 1);
  sq_alpha ~ normal(0, 50);
  sigma ~ inv_gamma(1, 1);

  // likelihood computation
  for (i in 1:nd) {
    int sub_n = count_decisions(d, i, n);
    vector[sub_n] mean_f = rep_vector(0, sub_n);
    matrix[sub_n, k] x_sub = sub_covariate_matrix(x, d, i, n, k);
    vector[sub_n] y_sub = sub_outcome_vector(y, d, i, n);
    matrix[sub_n, sub_n] L_K;
    matrix[sub_n, sub_n] K;
    K = cov_exp_quad_multidim(sq_alpha[i], rho[i], x_sub, sub_n);
    // diagonal elements
    for (j in 1:sub_n)
      K[j, j] = K[j, j] + sigma[i];
    L_K = cholesky_decompose(K);
    y_sub ~ multi_normal_cholesky(mean_f, L_K);
  }
}

generated quantities {
  matrix[ntest, nd] mu_test;
  matrix[cn, nd] mu_mat;
  vector[cn] mu;
  vector[cn] py;

  for(i in 1:nd) {
    int sub_n = count_decisions(d, i, n);
    matrix[sub_n, k] x_sub = sub_covariate_matrix(x, d, i, n, k);
    vector[sub_n] y_sub = sub_outcome_vector(y, d, i, n);
    mu_test[:, i] = gp_combine_rng(xtest, y_sub, x_sub, sq_alpha[i], rho[i], sigma[i], delta);
    mu_mat[:, i] = gp_combine_rng(cx, y_sub, x_sub, sq_alpha[i], rho[i], sigma[i], delta);
  }

  for(i in 1:cn) {
    mu[i] = mu_mat[i, cd[i]];
    py[i] = normal_rng(mu[i], sigma[cd[i]]);
  }

}
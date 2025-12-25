#include <RcppArmadillo.h>
#include <cmath>
#include <RcppDist.h>
//Code from RcppTN: https://github.com/olmjo/RcppTN/blob/master/src/rtn1.cpp
#include "rtn1.h"
// #include <omp.h> [[Rcpp::plugins(openmp)]]

//[[Rcpp::depends(RcppArmadillo, RcppDist)]]

using namespace Rcpp;
using namespace arma;
using namespace std;

extern "C" void sadmvn_(
    int* n, double *lower, double *upper, int* infin,
    double* corr, int* maxpts, double *abseps, double *releps,
    double *error, double *value, int *inform);

double log_sum_exp(double a, double b) {
  return(max(a, b) + log1p(exp(min(a, b) - max(a, b))));
}

double log_1p2exp(double a, double b) {
  double max_val = max(a, b);
  double min_val = min(a, b);
  if (max_val > 700) {
    return(max_val + log1p(exp(min_val - max_val) + exp(-max_val)));
  }
  return(log1p(exp(max_val) + exp(min_val)));
}

double clean_prob(double prob) {
  if (prob < 1e-9) {
    return(1e-9);
  }
  if (prob > (1 - 1e-9)) {
    return(1 - 1e-9);
  }
  return(prob);
}

double sample_ordinal_utility_probit_beta(
    mat y_star_m_lower, mat y_star_m_upper,
    mat alpha_v_1, mat alpha_v_2,
    mat delta_v_1, mat delta_v_2,
    double beta_mean, double beta_s) {

  y_star_m_lower += alpha_v_1 % delta_v_1;
  y_star_m_upper += alpha_v_2 % delta_v_2;

  double post_var = 1.0 / pow(beta_s, 2);
  double post_mean = beta_mean / pow(beta_s, 2);

  for (int j = 0; j < alpha_v_1.n_rows; j++) {
    post_var += dot(alpha_v_1.row(j), alpha_v_1.row(j)) +
      dot(alpha_v_2.row(j), alpha_v_2.row(j));
    post_mean += dot(alpha_v_1.row(j), y_star_m_lower.row(j)) +
      dot(alpha_v_2.row(j), y_star_m_upper.row(j));
  }
  return(randn() / sqrt(post_var) + post_mean / post_var);
}

//Inspired by stackover flow comment on pmvnorm
vec get_corr_v(mat X) {
  int n = X.n_cols;
  vec out_v(n * (n - 1) / 2);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      out_v(j + i * (i - 1) / 2) = X(i, j);
    }
  }
  return(out_v);
}

double sample_match_var_mvtnorm(
    vec alpha_post_mean_v, mat alpha_post_cov_s,
    vec delta_v, vec delta_mean_v, mat delta_cov_s) {

  int n = alpha_post_mean_v.n_elem / 2;
  // int nu = 0;
  int maxpts = 25000;
  double releps = 0;
  double abseps = 1e-6;
  // int rnd = 1;

  int* lower_int_ = new int[n];
  int* upper_int_ = new int[n];
  double* delta_ = new double[n];
  double* lower_alpha_post_mean_v_ = new double[n];
  double* upper_alpha_post_mean_v_ = new double[n];
  double* zero_v_ = new double[n];
  for (int i = 0; i < n; i++) {

    vec bound = -alpha_post_mean_v / sqrt(alpha_post_cov_s.diag());
    lower_alpha_post_mean_v_[i] = bound(i);
    upper_alpha_post_mean_v_[i] = bound(i + alpha_post_mean_v.n_elem / 2);

    lower_int_[i] = 0;
    upper_int_[i] = 1;
    delta_[i] = 0.0;
    zero_v_[i] = 0.0;
  }

  double* lower_corr_v_ = new double[n * (n - 1) / 2];
  double* upper_corr_v_ = new double[n * (n - 1) / 2];
  {
    mat alpha_post_corr_m =
      sqrt(diagmat(1 / alpha_post_cov_s.diag())) * alpha_post_cov_s *
      sqrt(diagmat(1 / alpha_post_cov_s.diag()));


    vec tmp_1 = get_corr_v(
      alpha_post_corr_m(span(0, alpha_post_mean_v.n_elem / 2 - 1),
                        span(0, alpha_post_mean_v.n_elem / 2 - 1)));
    vec tmp_2 = get_corr_v(
      alpha_post_corr_m(span(alpha_post_mean_v.n_elem / 2,
                             alpha_post_mean_v.n_elem - 1),
                             span(alpha_post_mean_v.n_elem / 2,
                                  alpha_post_mean_v.n_elem - 1)));
    for (int i = 0; i < tmp_1.n_elem; i++) {
      lower_corr_v_[i] = tmp_1(i);
      upper_corr_v_[i] = tmp_2(i);
    }
  }

  double err;
  double val;
  int inform;

  double sample_order_up_prob = as_scalar(
    dmvnorm(delta_v.t(), delta_mean_v, delta_cov_s, true));

  sadmvn_(&n, zero_v_, lower_alpha_post_mean_v_,
          lower_int_, lower_corr_v_,
          &maxpts, &abseps, &releps,
          &err, &val, &inform);
  sample_order_up_prob += log(val);

  sadmvn_(&n, upper_alpha_post_mean_v_, zero_v_,
          upper_int_, upper_corr_v_,
          &maxpts, &abseps, &releps,
          &err, &val, &inform);
  sample_order_up_prob += log(val);

  double sample_order_down_prob =
    as_scalar(dmvnorm(delta_v.t(), -delta_mean_v, delta_cov_s, true));
  sadmvn_(&n, lower_alpha_post_mean_v_, zero_v_,
          upper_int_, lower_corr_v_,
          &maxpts, &abseps, &releps,
          &err, &val, &inform);
  sample_order_down_prob += log(val);

  sadmvn_(&n, zero_v_, upper_alpha_post_mean_v_,
          lower_int_, upper_corr_v_,
          &maxpts, &abseps, &releps,
          &err, &val, &inform);
  sample_order_down_prob += log(val);

  delete [] lower_int_;
  delete [] upper_int_;
  delete [] delta_;
  delete [] lower_alpha_post_mean_v_;
  delete [] upper_alpha_post_mean_v_;
  delete [] zero_v_;
  delete [] lower_corr_v_;
  delete [] upper_corr_v_;

  if (!is_finite(sample_order_down_prob) &&
      !is_finite(sample_order_up_prob)) {
      sample_order_up_prob = 0;
    sample_order_down_prob = 0;
  }
  double log_sample_prob = sample_order_up_prob -
    (max(sample_order_up_prob, sample_order_down_prob) +
    log(1 + exp(min(sample_order_up_prob, sample_order_down_prob) -
    max(sample_order_up_prob, sample_order_down_prob))));
  double match_var = (log(randu()) < log_sample_prob) * 2 - 1;

  return(match_var);
}

// [[Rcpp::export]]
arma::vec sample_alpha_ordinal_independent_lower(
    arma::vec alpha_post_mean_m, arma::mat alpha_post_cov_s, int num_iter = 20) {

  vec out_v(alpha_post_mean_m.n_elem);
  out_v(0) = rtn1(alpha_post_mean_m(0), sqrt(alpha_post_cov_s(0, 0)),
      -datum::inf, 0);
  for (int m = 1; m < alpha_post_mean_m.n_elem / 2; m++) {
    out_v(m) = rtn1(alpha_post_mean_m(m), sqrt(alpha_post_cov_s(m, m)),
          out_v(m - 1), 0);
  }

  for (int i = 0; i < num_iter; i++) {
    out_v(0) = rtn1(alpha_post_mean_m(0), sqrt(alpha_post_cov_s(0, 0)),
                    -datum::inf, out_v(1));
    for (int m = 1; m < alpha_post_mean_m.n_elem / 2 - 1; m++) {
      out_v(m) = rtn1(alpha_post_mean_m(m), sqrt(alpha_post_cov_s(m, m)),
            out_v(m - 1), out_v(m + 1));
    }
    out_v(alpha_post_mean_m.n_elem / 2 - 1) =
      rtn1(alpha_post_mean_m(alpha_post_mean_m.n_elem / 2 - 1),
           sqrt(alpha_post_cov_s(alpha_post_mean_m.n_elem / 2 - 1,
                                 alpha_post_mean_m.n_elem / 2 - 1)),
           out_v(alpha_post_mean_m.n_elem / 2 - 2), 0);
  }

  out_v(alpha_post_mean_m.n_elem / 2) =
    rtn1(alpha_post_mean_m(alpha_post_mean_m.n_elem / 2),
         sqrt(alpha_post_cov_s(alpha_post_mean_m.n_elem / 2, alpha_post_mean_m.n_elem / 2)),
         0, datum::inf);
  for (int m = alpha_post_mean_m.n_elem / 2 + 1; m < alpha_post_mean_m.n_elem; m++) {
    out_v(m) = rtn1(alpha_post_mean_m(m), sqrt(alpha_post_cov_s(m, m)),
          out_v(m - 1), datum::inf);
  }
  for (int i = 0; i < num_iter; i++) {
    out_v(alpha_post_mean_m.n_elem / 2) =
      rtn1(alpha_post_mean_m(alpha_post_mean_m.n_elem / 2),
          sqrt(alpha_post_cov_s(alpha_post_mean_m.n_elem / 2,
                                alpha_post_mean_m.n_elem / 2)),
          0, out_v(alpha_post_mean_m.n_elem / 2 + 1));
    for (int m = alpha_post_mean_m.n_elem / 2 + 1;
         m < alpha_post_mean_m.n_elem - 1; m++) {
      out_v(m) = rtn1(alpha_post_mean_m(m), sqrt(alpha_post_cov_s(m, m)),
            out_v(m - 1), out_v(m + 1));
    }
    out_v(alpha_post_mean_m.n_elem - 1) =
      rtn1(alpha_post_mean_m(alpha_post_mean_m.n_elem - 1),
           sqrt(alpha_post_cov_s(alpha_post_mean_m.n_elem - 1,
                                 alpha_post_mean_m.n_elem - 1)),
           out_v(alpha_post_mean_m.n_elem - 2), datum::inf);
  }
  return(out_v);
}

vec sample_alpha_ordinal_independent_upper(
    vec alpha_post_mean_m, mat alpha_post_cov_s, int num_iter = 20) {

  vec out_v(alpha_post_mean_m.n_elem);
  out_v(alpha_post_mean_m.n_elem - 1) =
    rtn1(alpha_post_mean_m(alpha_post_mean_m.n_elem - 1),
         sqrt(alpha_post_cov_s(alpha_post_mean_m.n_elem - 1,
                               alpha_post_mean_m.n_elem - 1)),
        -datum::inf, 0);
  for (int m = alpha_post_mean_m.n_elem - 2;
        m > alpha_post_mean_m.n_elem / 2 - 1; m--) {
    out_v(m) = rtn1(alpha_post_mean_m(m), sqrt(alpha_post_cov_s(m, m)),
          out_v(m + 1), 0);
  }
  for (int i = 0; i < num_iter; i++) {
    out_v(alpha_post_mean_m.n_elem - 1) =
      rtn1(alpha_post_mean_m(alpha_post_mean_m.n_elem - 1),
           sqrt(alpha_post_cov_s(alpha_post_mean_m.n_elem - 1,
                                 alpha_post_mean_m.n_elem - 1)),
            -datum::inf, out_v(alpha_post_mean_m.n_elem - 2));
    for (int m = alpha_post_mean_m.n_elem - 2;
         m > alpha_post_mean_m.n_elem / 2; m--) {
      out_v(m) = rtn1(alpha_post_mean_m(m), sqrt(alpha_post_cov_s(m, m)),
            out_v(m + 1), out_v(m - 1));
    }
    out_v(alpha_post_mean_m.n_elem / 2) =
      rtn1(alpha_post_mean_m(alpha_post_mean_m.n_elem / 2),
           sqrt(alpha_post_cov_s(alpha_post_mean_m.n_elem / 2,
                                 alpha_post_mean_m.n_elem / 2)),
            out_v(alpha_post_mean_m.n_elem / 2 + 1), 0);
  }

  out_v(alpha_post_mean_m.n_elem / 2 - 1) =
    rtn1(alpha_post_mean_m(alpha_post_mean_m.n_elem / 2 - 1),
         sqrt(alpha_post_cov_s(alpha_post_mean_m.n_elem / 2 - 1,
                               alpha_post_mean_m.n_elem / 2 - 1)),
         0, datum::inf);
  for (int m = alpha_post_mean_m.n_elem / 2 - 2; m > -1; m--) {
    out_v(m) = rtn1(alpha_post_mean_m(m), sqrt(alpha_post_cov_s(m, m)),
          out_v(m + 1), datum::inf);
  }
  for (int i = 0; i < num_iter; i++) {
    out_v(alpha_post_mean_m.n_elem / 2 - 1) =
      rtn1(alpha_post_mean_m(alpha_post_mean_m.n_elem / 2 - 1),
           sqrt(alpha_post_cov_s(alpha_post_mean_m.n_elem / 2 - 1,
                                 alpha_post_mean_m.n_elem / 2 - 1)),
                                 0, out_v(alpha_post_mean_m.n_elem / 2 - 2));
    for (int m = alpha_post_mean_m.n_elem / 2 - 2; m > 0; m--) {
      out_v(m) = rtn1(alpha_post_mean_m(m), sqrt(alpha_post_cov_s(m, m)),
            out_v(m + 1), out_v(m - 1));
    }
    out_v(0) =
      rtn1(alpha_post_mean_m(0), sqrt(alpha_post_cov_s(0, 0)),
           out_v(1), datum::inf);
  }
  return(out_v);
}

vec sample_ordinal_probit_matched_alpha_intervals(
    mat y_star_m, vec beta_v, vec delta_v,
    vec alpha_mean_v, mat alpha_cov_s,
    vec delta_mean_v, mat delta_cov_s) {


  mat post_cov = alpha_cov_s.i();
  vec post_mean = solve(alpha_cov_s, alpha_mean_v);

  for (int m = 0; m < alpha_mean_v.n_elem; m++) {
    post_cov(m, m) += dot(
      beta_v - delta_v(m), beta_v - delta_v(m));
    post_mean(m) += dot(
      beta_v - delta_v(m), y_star_m.col(m));
  }
  post_mean = solve(post_cov, post_mean);
  post_cov = post_cov.i();

  mat trans_m = eye(post_mean.n_elem, post_mean.n_elem);
  trans_m(span(0, alpha_mean_v.n_elem / 2 - 1),
          span(0, alpha_mean_v.n_elem / 2 - 1)) +=
    diagmat(-ones(alpha_mean_v.n_elem / 2 - 1), 1);
  trans_m(span(alpha_mean_v.n_elem / 2,
               alpha_mean_v.n_elem - 1),
          span(alpha_mean_v.n_elem / 2,
               alpha_mean_v.n_elem - 1)) +=
    diagmat(-ones(alpha_mean_v.n_elem / 2 - 1), -1);
  vec int_post_mean = trans_m * post_mean;
  mat int_post_cov = trans_m * post_cov * trans_m.t();


  vec output_v(post_mean.n_elem + 1);
  output_v(post_mean.n_elem) = sample_match_var_mvtnorm(
    int_post_mean, int_post_cov, delta_v, delta_mean_v, delta_cov_s);

  if (output_v(post_mean.n_elem) > 0) {
    output_v(span(0, post_mean.n_elem - 1)) =
      sample_alpha_ordinal_independent_lower(
        post_mean, post_cov);
  } else {
    output_v(span(0, post_mean.n_elem - 1)) =
      sample_alpha_ordinal_independent_upper(
        post_mean, post_cov);
  }

  return(output_v);
}

vec sample_ordinal_probit_matched_alpha_intervals_no_flip(
    mat y_star_m, vec beta_v, vec delta_v,
    vec alpha_mean_v, mat alpha_cov_s,
    vec delta_mean_v, mat delta_cov_s) {


  mat post_cov = alpha_cov_s.i();
  vec post_mean = solve(alpha_cov_s, alpha_mean_v);

  for (int m = 0; m < alpha_mean_v.n_elem; m++) {
    post_cov(m, m) += dot(
      beta_v - delta_v(m), beta_v - delta_v(m));
    post_mean(m) += dot(
      beta_v - delta_v(m), y_star_m.col(m));
  }
  post_mean = solve(post_cov, post_mean);
  post_cov = post_cov.i();

  mat trans_m = eye(post_mean.n_elem, post_mean.n_elem);
  trans_m(span(0, alpha_mean_v.n_elem / 2 - 1),
          span(0, alpha_mean_v.n_elem / 2 - 1)) +=
            diagmat(-ones(alpha_mean_v.n_elem / 2 - 1), 1);
  trans_m(span(alpha_mean_v.n_elem / 2,
               alpha_mean_v.n_elem - 1),
               span(alpha_mean_v.n_elem / 2,
                    alpha_mean_v.n_elem - 1)) +=
                      diagmat(-ones(alpha_mean_v.n_elem / 2 - 1), -1);
  vec int_post_mean = trans_m * post_mean;
  mat int_post_cov = trans_m * post_cov * trans_m.t();


  vec output_v(post_mean.n_elem + 1);
  output_v(post_mean.n_elem) = 1;

  output_v(span(0, post_mean.n_elem - 1)) =
    sample_alpha_ordinal_independent_lower(
      post_mean, post_cov);


  return(output_v);
}

vec sample_ordinal_utility_matched_delta(
    mat y_star_m_lower, mat y_star_m_upper,
    vec alpha_v_lower, vec alpha_v_upper,
    vec beta_v, double match_var,
    vec delta_mean_v, mat delta_cov_s) {

  y_star_m_lower -= resize(beta_v * alpha_v_lower.t(), size(y_star_m_lower));
  y_star_m_upper -= resize(beta_v * alpha_v_upper.t(), size(y_star_m_upper));
  y_star_m_lower *= -1;
  y_star_m_upper *= -1;
  mat post_cov = beta_v.n_elem *
    diagmat(join_vert(alpha_v_lower, alpha_v_upper) %
              join_vert(alpha_v_lower, alpha_v_upper)) +
    delta_cov_s.i();
  vec post_mean = match_var * solve(delta_cov_s, delta_mean_v);
  for (int m = 0; m < alpha_v_lower.n_elem; m++) {
    post_mean(m) += accu(alpha_v_lower(m) * y_star_m_lower.col(m));
    post_mean(m + alpha_v_lower.n_elem) +=
      accu(alpha_v_upper(m) * y_star_m_upper.col(m));
  }
  return(rmvnorm(1, solve(post_cov, post_mean),
                 post_cov.i()).t());
}

// [[Rcpp::export]]
double calc_choice_k_prob(
  arma::vec mean_1, arma::vec mean_2, int choice_k) {

  int n = 2 * mean_1.n_elem;
  // int nu = 0;
  int maxpts = 25000;
  double releps = 0;
  double abseps = 1e-6;
  // int rnd = 1;
  double err;
  double val;
  int inform;

  int* upper_int_ = new int[n];
  double* delta_ = new double[n];
  double* post_mean_v_ = new double[n];
  double* post_mean_v_2_ = new double[n];
  double* zero_v_ = new double[n];

  for (int i = 0; i < n; i++) {
    upper_int_[i] = 1;
    delta_[i] = 0.0;
    zero_v_[i] = 0.0;
  }

  double* corr_v_ = new double[n * (n - 1) / 2];
  for (int i = 0; i < n * (n - 1) / 2; i++) {
    corr_v_[i] = 0.5;
  }

  double yea_prob;
  if (choice_k != mean_1.n_elem) {
    for (int i = 0; i < mean_1.n_elem; i++) {
      post_mean_v_[i] = (mean_1(i) - mean_1(choice_k)) / sqrt(2.0);
      post_mean_v_[i + mean_1.n_elem] = (mean_2(i) - mean_1(choice_k)) / sqrt(2.0);

      post_mean_v_2_[i] = (mean_1(i) - mean_2(mean_2.n_elem - 1 - choice_k)) / sqrt(2.0);
      post_mean_v_2_[i + mean_1.n_elem] = (mean_2(i) - mean_2(mean_2.n_elem - 1 - choice_k)) / sqrt(2.0);
    }
    post_mean_v_[choice_k] = -mean_1(choice_k) / sqrt(2.0);
    post_mean_v_2_[n - choice_k - 1] = -mean_2(mean_2.n_elem - 1 - choice_k) / sqrt(2.0);

    sadmvn_(&n, post_mean_v_, zero_v_, upper_int_, corr_v_,
            &maxpts, &abseps, &releps, &err, &val, &inform);
    yea_prob = val;

    sadmvn_(&n, post_mean_v_2_, zero_v_, upper_int_, corr_v_,
            &maxpts, &abseps, &releps, &err, &val, &inform);
    yea_prob += val;
  } else {
    for (int i = 0; i < mean_1.n_elem; i++) {
      post_mean_v_[i] = mean_1(i) / sqrt(2.0);
      post_mean_v_[i + mean_1.n_elem] = mean_2(i) / sqrt(2.0);
    }
    sadmvn_(&n, post_mean_v_, zero_v_, upper_int_, corr_v_,
            &maxpts, &abseps, &releps, &err, &val, &inform);
    yea_prob = val;

  }

  delete [] upper_int_;
  delete [] delta_;
  delete [] post_mean_v_;
  delete [] post_mean_v_2_;
  delete [] zero_v_;
  delete [] corr_v_;

  return(yea_prob);
}

vec flip_signs(
    uvec vote, vec alpha_v_lower, vec alpha_v_upper, vec beta_v,
    vec delta_v_lower, vec delta_v_upper) {

  double curr_prob = 0;
  double flip_prob = 0;

  for (int i = 0; i < beta_v.n_elem; i++) {
    vec mean_v_lower = alpha_v_lower % (beta_v(i) - delta_v_lower);
    vec mean_v_upper = alpha_v_upper % (beta_v(i) - delta_v_upper);

    curr_prob += log(calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i)));

    mean_v_lower = -alpha_v_lower % (beta_v(i) + delta_v_lower);
    mean_v_upper = -alpha_v_upper % (beta_v(i) + delta_v_upper);

    flip_prob += log(calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i)));
  }

  if (log(randu()) < flip_prob - curr_prob) {
    return(-join_vert(alpha_v_lower, alpha_v_upper,
                      delta_v_lower, delta_v_upper));
  }
  return(join_vert(alpha_v_lower, alpha_v_upper,
                  delta_v_lower, delta_v_upper));
}

vec flip_signs_prior(
    uvec vote, vec alpha_v_lower, vec alpha_v_upper, vec beta_v,
    vec delta_v_lower, vec delta_v_upper, double sign,
    vec alpha_mean_v, mat alpha_sigma_m,
    vec delta_mean_v, mat delta_sigma_m) {

  double curr_prob = 0;
  double flip_prob = 0;

  vec flip_alpha_v_lower;
  vec flip_alpha_v_upper;
  vec flip_delta_v_lower;
  vec flip_delta_v_upper;

  {
    vec flip_alpha_v;

    if (sign < 0) {
      flip_alpha_v =
        sample_alpha_ordinal_independent_lower(alpha_mean_v, alpha_sigma_m);
    } else {
      flip_alpha_v =
        sample_alpha_ordinal_independent_upper(alpha_mean_v, alpha_sigma_m);
    }
    flip_alpha_v_lower = flip_alpha_v(span(0, alpha_v_lower.n_elem - 1));
    flip_alpha_v_upper = flip_alpha_v(
      span(alpha_v_lower.n_elem, 2 * alpha_v_lower.n_elem - 1));

    vec flip_delta_v =
      rmvnorm(1, -sign * delta_mean_v, delta_sigma_m).t();
    flip_delta_v_lower = flip_delta_v(span(0, delta_v_lower.n_elem - 1));
    flip_delta_v_upper = flip_delta_v(
      span(delta_v_lower.n_elem, 2 * delta_v_lower.n_elem - 1));
  }

  for (int i = 0; i < beta_v.n_elem; i++) {
    vec mean_v_lower = alpha_v_lower % (beta_v(i) - delta_v_lower);
    vec mean_v_upper = alpha_v_upper % (beta_v(i) - delta_v_upper);

    curr_prob += log(calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i)));

    mean_v_lower = flip_alpha_v_lower % (beta_v(i) - flip_delta_v_lower);
    mean_v_upper = flip_alpha_v_upper % (beta_v(i) - flip_delta_v_upper);

    flip_prob += log(calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i)));
  }

  if (log(randu()) < flip_prob - curr_prob) {
    return(join_vert(flip_alpha_v_lower, flip_alpha_v_upper,
                     flip_delta_v_lower, flip_delta_v_upper));
  }
  return(join_vert(alpha_v_lower, alpha_v_upper,
                   delta_v_lower, delta_v_upper));
}

vec flip_signs_response(
    uvec vote, vec alpha_v_lower, vec alpha_v_upper, vec beta_v,
    vec delta_v_lower, vec delta_v_upper,
    vec alpha_mean_v, mat alpha_sigma_m,
    vec delta_mean_v, mat delta_sigma_m) {

  double curr_prob = 0;
  double flip_prob = 0;

  vec flip_alpha_v_lower;
  vec flip_alpha_v_upper;
  vec flip_delta_v_lower;
  vec flip_delta_v_upper;

  {
    vec flip_alpha_v;

    flip_alpha_v =
      sample_alpha_ordinal_independent_lower(alpha_mean_v, alpha_sigma_m);
    flip_alpha_v_lower = flip_alpha_v(span(0, alpha_v_lower.n_elem - 1));
    flip_alpha_v_upper = flip_alpha_v(
      span(alpha_v_lower.n_elem, 2 * alpha_v_lower.n_elem - 1));

    vec flip_delta_v =
      rmvnorm(1, delta_mean_v, delta_sigma_m).t();
    flip_delta_v_lower = flip_delta_v(span(0, delta_v_lower.n_elem - 1));
    flip_delta_v_upper = flip_delta_v(
      span(delta_v_lower.n_elem, 2 * delta_v_lower.n_elem - 1));
  }

  for (int i = 0; i < beta_v.n_elem; i++) {
    vec mean_v_lower = alpha_v_lower % (beta_v(i) - delta_v_lower);
    vec mean_v_upper = alpha_v_upper % (beta_v(i) - delta_v_upper);

    curr_prob += log(calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i)));

    mean_v_lower = flip_alpha_v_lower % (beta_v(i) - flip_delta_v_lower);
    mean_v_upper = flip_alpha_v_upper % (beta_v(i) - flip_delta_v_upper);

    double prob = calc_choice_k_prob(mean_v_lower, mean_v_upper, alpha_v_lower.n_elem - vote(i));
    if (prob < 1e-9) {
      flip_prob += log(1e-9);
    } else {
      flip_prob += log(prob);
    }
    // flip_prob += log(calc_choice_k_prob(mean_v_lower, mean_v_upper, alpha_v_lower.n_elem - vote(i)));
  }

  if (log(randu()) < flip_prob - curr_prob) {
    vec tmp = {1};
    return(join_vert(join_vert(flip_alpha_v_lower, flip_alpha_v_upper,
                               flip_delta_v_lower, flip_delta_v_upper), tmp));
  }
  vec tmp = {0};
  return(join_vert(join_vert(alpha_v_lower, alpha_v_upper,
                             delta_v_lower, delta_v_upper),
                   tmp));
}

double flip_signs_beta(
    uvec vote, mat alpha_v_lower, mat alpha_v_upper, double beta_val,
    mat delta_v_lower, mat delta_v_upper,
    uvec km1,
    double beta_mean, double beta_sd, double proposal_sd) {

  double flip_beta_val = randn() * proposal_sd - beta_val;

  double curr_prob = R::dnorm(beta_val, beta_mean, beta_sd, true);
  double flip_prob = R::dnorm(flip_beta_val, beta_mean, beta_sd, true);;

  for (unsigned int i = 0; i < vote.n_elem; i++) {

    unsigned int half_alpha_size = km1(i);
    uvec interested_col = {i};
    uvec interested_rows_lower = linspace<uvec>(0, half_alpha_size - 1, half_alpha_size);
    uvec interested_rows_upper =
      linspace<uvec>(alpha_v_lower.n_rows - half_alpha_size,
                     alpha_v_lower.n_rows - 1, half_alpha_size);

    vec alpha_v_lower_tmp =
      alpha_v_lower(interested_rows_lower, interested_col);
    vec alpha_v_upper_tmp =
      alpha_v_upper(interested_rows_upper, interested_col);
    vec delta_v_lower_tmp =
      delta_v_lower(interested_rows_lower, interested_col);
    vec delta_v_upper_tmp =
      delta_v_upper(interested_rows_upper, interested_col);

    vec mean_v_lower = alpha_v_lower_tmp % (beta_val - delta_v_lower_tmp);
    vec mean_v_upper = alpha_v_upper_tmp % (beta_val - delta_v_upper_tmp);

    curr_prob += log(clean_prob(calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i))));

    mean_v_lower = alpha_v_lower_tmp % (flip_beta_val - delta_v_lower_tmp);
    mean_v_upper = alpha_v_upper_tmp % (flip_beta_val - delta_v_upper_tmp);

    // double prob = calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i));
    // if (prob < 1e-9) {
    //   flip_prob += log(1e-9);
    // } else {
    //   flip_prob += log(prob);
    // }
    flip_prob += log(clean_prob(calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i))));
  }

  if (log(randu()) < flip_prob - curr_prob) {
    // vec tmp = {1};
    return(flip_beta_val);
  }
  // vec tmp = {0};
  return(beta_val);
}

List flip_signs_beta_parallel(
    uvec vote, mat alpha_v_lower, mat alpha_v_upper, vec beta_v,
    mat delta_v_lower, mat delta_v_upper,
    double beta_mean, double beta_sd, double proposal_sd,
    uvec respondent_v, uvec unique_respondent_v,
    uvec question_v, uvec question_num_choices_m1_v,
    unsigned int num_respondent, unsigned int pos_ind, unsigned int neg_ind,
    int cores = 1) {

  vec flip_beta_v = randn(beta_v.n_elem) * proposal_sd - beta_v;
  vec curr_log_ll(vote.n_elem);
  vec flip_log_ll(vote.n_elem);
  vec accept_v(beta_v.n_elem, fill::zeros);

  unsigned int half_alpha_size, q_num, ind_num;
  double beta_val, flip_beta_val;
  vec mean_v_lower, mean_v_upper, alpha_v_lower_tmp, alpha_v_upper_tmp,
    delta_v_lower_tmp, delta_v_upper_tmp;
  uvec interested_col, interested_rows_lower, interested_rows_upper;

  // omp_set_num_threads(cores);
  /* #pragma omp parallel for schedule(static) private( \
    half_alpha_size, q_num, ind_num, beta_val, \
    flip_beta_val, mean_v_lower, mean_v_upper, \
    alpha_v_lower_tmp, alpha_v_upper_tmp, \
    delta_v_lower_tmp, delta_v_upper_tmp, \
    interested_col, interested_rows_lower, interested_rows_upper)
  */
  for (unsigned int i = 0; i < vote.n_elem; i++) {

    q_num = question_v(i);
    ind_num = respondent_v(i);
    half_alpha_size = question_num_choices_m1_v(q_num);
    beta_val = beta_v(ind_num);
    flip_beta_val = flip_beta_v(ind_num);

    interested_col = {q_num};
    interested_rows_lower = linspace<uvec>(0, half_alpha_size - 1, half_alpha_size);
    interested_rows_upper =
      linspace<uvec>(alpha_v_lower.n_rows - half_alpha_size,
                     alpha_v_lower.n_rows - 1, half_alpha_size);

    alpha_v_lower_tmp =
      alpha_v_lower(interested_rows_lower, interested_col);
    alpha_v_upper_tmp =
      alpha_v_upper(interested_rows_upper, interested_col);
    delta_v_lower_tmp =
      delta_v_lower(interested_rows_lower, interested_col);
    delta_v_upper_tmp =
      delta_v_upper(interested_rows_upper, interested_col);

    mean_v_lower = alpha_v_lower_tmp % (beta_val - delta_v_lower_tmp);
    mean_v_upper = alpha_v_upper_tmp % (beta_val - delta_v_upper_tmp);

    curr_log_ll(i) = log(clean_prob(calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i))));

    mean_v_lower = alpha_v_lower_tmp % (flip_beta_val - delta_v_lower_tmp);
    mean_v_upper = alpha_v_upper_tmp % (flip_beta_val - delta_v_upper_tmp);

    // double prob = calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i));
    // if (prob < 1e-9) {
    //   flip_prob += log(1e-9);
    // } else {
    //   flip_prob += log(prob);
    // }
    flip_log_ll(i) = log(clean_prob(calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i))));
  }

  vec return_beta_v = beta_v;
  vec log_ll = curr_log_ll;
  for (unsigned int i : unique_respondent_v) {

    if (i == pos_ind || i == neg_ind) {
      continue;
    }

    uvec interested_inds = find(respondent_v == i);
    double curr_prob =
      R::dnorm(beta_v(i), beta_mean, beta_sd, true) +
        accu(curr_log_ll(interested_inds));
    double flip_prob =
      R::dnorm(flip_beta_v(i), beta_mean, beta_sd, true) +
        accu(flip_log_ll(interested_inds));

    if (log(randu()) < flip_prob - curr_prob) {
      return_beta_v(i) = flip_beta_v(i);
      log_ll(interested_inds) = flip_log_ll(interested_inds);
      accept_v(i) = 1;
    }
  }

  return(List::create(return_beta_v, log_ll, accept_v));
}

List flip_signs_beta_parallel_2(
    uvec vote, vec alpha_v_lower, vec alpha_v_upper, vec beta_v,
    vec delta_v_lower, vec delta_v_upper,
    double beta_mean, double beta_sd, double proposal_sd,
    uvec respondent_v, uvec unique_respondent_v,
    uvec question_v, uvec question_num_choices_m1_v,
    unsigned int max_response_num, unsigned int pos_ind, unsigned int neg_ind,
    int cores = 1) {

  vec flip_beta_v = randn(beta_v.n_elem) * proposal_sd - beta_v;
  vec curr_log_ll(vote.n_elem);
  vec flip_log_ll(vote.n_elem);
  vec accept_v(beta_v.n_elem, fill::zeros);

  unsigned int half_alpha_size, q_num, ind_num;
  double beta_val, flip_beta_val;
  vec mean_v_lower, mean_v_upper, alpha_v_lower_tmp, alpha_v_upper_tmp,
  delta_v_lower_tmp, delta_v_upper_tmp;
  uvec interested_col, interested_rows_lower, interested_rows_upper;

  // omp_set_num_threads(cores);
  /*
#pragma omp parallel for schedule(static) private( \
  half_alpha_size, q_num, ind_num, beta_val,       \
  flip_beta_val, mean_v_lower, mean_v_upper,       \
  alpha_v_lower_tmp, alpha_v_upper_tmp,            \
  delta_v_lower_tmp, delta_v_upper_tmp,            \
  interested_col, interested_rows_lower, interested_rows_upper)
   */
    for (unsigned int i = 0; i < vote.n_elem; i++) {

      q_num = question_v(i);
      ind_num = respondent_v(i);
      half_alpha_size = question_num_choices_m1_v(q_num);
      beta_val = beta_v(ind_num);
      flip_beta_val = flip_beta_v(ind_num);

      interested_col = {q_num};
      interested_rows_lower = linspace<uvec>(0, half_alpha_size - 1, half_alpha_size);
      interested_rows_upper =
        linspace<uvec>(max_response_num - half_alpha_size,
                       max_response_num - 1, half_alpha_size);

      alpha_v_lower_tmp =
        alpha_v_lower(interested_rows_lower + q_num * max_response_num);
      alpha_v_upper_tmp =
        alpha_v_upper(interested_rows_upper + q_num * max_response_num);
      delta_v_lower_tmp =
        delta_v_lower(interested_rows_lower + q_num * max_response_num);
      delta_v_upper_tmp =
        delta_v_upper(interested_rows_upper + q_num * max_response_num);

      mean_v_lower = alpha_v_lower_tmp % (beta_val - delta_v_lower_tmp);
      mean_v_upper = alpha_v_upper_tmp % (beta_val - delta_v_upper_tmp);

      int vote_choice = vote(i);

      double curr_tmp = log(clean_prob(calc_choice_k_prob(mean_v_lower, mean_v_upper, vote_choice)));

      mean_v_lower = alpha_v_lower_tmp % (flip_beta_val - delta_v_lower_tmp);
      mean_v_upper = alpha_v_upper_tmp % (flip_beta_val - delta_v_upper_tmp);

      double flip_tmp = log(clean_prob(calc_choice_k_prob(mean_v_lower, mean_v_upper, vote_choice)));

      #pragma omp atomic write
      curr_log_ll(i) = curr_tmp;

      #pragma omp atomic write
      flip_log_ll(i) = flip_tmp;

      // curr_log_ll(i) = log(clean_prob(calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i))));
      //
      // mean_v_lower = alpha_v_lower_tmp % (flip_beta_val - delta_v_lower_tmp);
      // mean_v_upper = alpha_v_upper_tmp % (flip_beta_val - delta_v_upper_tmp);
      //
      // // double prob = calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i));
      // // if (prob < 1e-9) {
      // //   flip_prob += log(1e-9);
      // // } else {
      // //   flip_prob += log(prob);
      // // }
      // flip_log_ll(i) = log(clean_prob(calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i))));
    }

    vec return_beta_v = beta_v;
  vec log_ll = curr_log_ll;
  for (unsigned int i : unique_respondent_v) {

    if (i == pos_ind || i == neg_ind) {
      continue;
    }

    uvec interested_inds = find(respondent_v == i);
    double curr_prob =
      R::dnorm(beta_v(i), beta_mean, beta_sd, true) +
      accu(curr_log_ll(interested_inds));
    double flip_prob =
      R::dnorm(flip_beta_v(i), beta_mean, beta_sd, true) +
      accu(flip_log_ll(interested_inds));

    if (log(randu()) < flip_prob - curr_prob) {
      return_beta_v(i) = flip_beta_v(i);
      log_ll(interested_inds) = flip_log_ll(interested_inds);
      accept_v(i) = 1;
    }
  }

  return(List::create(return_beta_v, log_ll, accept_v));
}

// List update_waic(vec new_log_ll, vec mean_prob, vec mean_log_prob,
//                 vec log_prob_var, num_iter) {
//
//   for (unsigned int i = 0; i < new_log_ll.n_elem; i++) {
//     mean_prob(i) = log_sum_exp(mean_prob(i), new_log_ll(i));
//   }
//   vec next_mean_log_prob =
//     (num_iter * mean_log_prob + new_log_ll) / (num_iter + 1);
//   log_prob_var +=
//     (new_log_ll - mean_log_prob) * (new_log_ll - next_mean_log_prob);
//   mean_log_prob = next_mean_log_prob;
//   return(List::create(mean_prob, mean_log_prob, log_prob_var));
// }

vec sample_y_star_m_na(vec mean_m_1, vec mean_m_2) {
  vec out_v(2 * mean_m_1.n_elem + 1, fill::randn);
  for (int k = 0; k < mean_m_1.n_elem; k++) {
    out_v(k) += mean_m_1(k);
    out_v(mean_m_1.n_elem + 1 + k) += mean_m_2(k);
  }
  return(out_v);
}

double sample_y_star_k(int k, int other_k, vec y_star, double mean_m_k) {
  bool check_pass = true;
  double max_val = -datum::inf;
  for (int m = 0; m < y_star.n_elem; m++) {
    if (m == k || m == other_k) {
      continue;
    }
    if (y_star(m) > max_val) {
      max_val = y_star(m);
    }
    if (y_star(m) > y_star(other_k)) {
      check_pass = false;
    }
  }
  if (check_pass) {
    return(randn() + mean_m_k);
  }
  return(rtn1(mean_m_k, 1, max_val, datum::inf));
}

double sample_y_star_m_K(int k, int K, vec y_star) {
  if (k == K) {
    double max_val = -datum::inf;
    for (int m = 0; m < y_star.n_elem; m++) {
      if (m == K) {
        continue;
      }
      if (y_star(m) > max_val) {
        max_val = y_star(m);
      }
    }
    return(
      rtn1(0, 1, max_val, datum::inf));
  } else {
    return(
      rtn1(0, 1, -datum::inf,
           max(y_star(k), y_star(y_star.n_elem - 1 - k))));
  }
}

vec sample_y_star_m_1_k(int k, vec y_star, vec mean_m_1) {

  for (int m = 0; m < mean_m_1.n_elem; m++) {
    if (m == k) {
      y_star(m) = sample_y_star_k(
        k, y_star.n_elem - 1 - k, y_star, mean_m_1(m));
    } else {
      y_star(m) = rtn1(mean_m_1(m), 1, -datum::inf,
        max(y_star(k), y_star(y_star.n_elem - 1 - k)));
    }
  }
  return(y_star);
}

vec sample_y_star_m_2_k(int k, vec y_star, vec mean_m_2) {

  for (int m = 0; m < mean_m_2.n_elem; m++) {
    int m_ind = m + mean_m_2.n_elem + 1;
    if (m_ind == 2 * mean_m_2.n_elem - k) {
      y_star(m_ind) = sample_y_star_k(
        m_ind, k, y_star, mean_m_2(m));
    } else {
      y_star(m_ind) = rtn1(mean_m_2(m), 1, -datum::inf,
        max(y_star(k), y_star(y_star.n_elem - 1 - k)));
    }
  }
  return(y_star);
}

vec sample_y_star_m(vec y_star_vec, int k, vec alpha_1, vec alpha_2,
                    double leg_pos, vec delta_1, vec delta_2) {

  vec out_vec(y_star_vec.n_elem);
  vec mean_m_1 = alpha_1 % (leg_pos - delta_1);
  vec mean_m_2 = alpha_2 % (leg_pos - delta_2);
  out_vec = sample_y_star_m_1_k(k, y_star_vec, mean_m_1);
  out_vec(alpha_1.n_elem) = sample_y_star_m_K(k, alpha_1.n_elem, out_vec);
  out_vec = sample_y_star_m_2_k(k, out_vec, mean_m_2);
  return(out_vec);
}

double calc_log_ll_y_star(
    uvec vote_v, mat y_star_m, mat alpha_v_lower, mat alpha_v_upper, vec beta_v,
    mat delta_v_lower, mat delta_v_upper,
    uvec respondent_v, uvec question_v, uvec question_num_choices_m1_v) {

  double log_prob = 0;
  for (unsigned int i = 0; i < y_star_m.n_rows; i++) {

    unsigned int q_num = question_v(i);
    unsigned int ind_num = respondent_v(i);
    unsigned int half_alpha_size = question_num_choices_m1_v(q_num);
    double beta_val = beta_v(ind_num);

    // uvec interested_col = {q_num};
    // uvec interested_rows_lower = linspace<uvec>(0, half_alpha_size - 1, half_alpha_size);
    // uvec interested_rows_upper =
    //   linspace<uvec>(alpha_v_lower.n_rows - half_alpha_size,
    //                  alpha_v_lower.n_rows - 1, half_alpha_size);

    vec mean_v(2 * half_alpha_size);
    double max_val, second_max_val;
    if (vote_v(i) != half_alpha_size) {
      max_val =
        max(y_star_m(i, vote_v(i)),
            y_star_m(i, y_star_m.n_cols - 1 - vote_v(i)));
      second_max_val = y_star_m(i, (y_star_m.n_cols - 1) / 2);
    } else {
      max_val = y_star_m(i, (y_star_m.n_cols - 1) / 2);
      second_max_val = -datum::inf;
    }

    // double max_val = max(y_star_m(vote_v(i)),
    //                      y_star_m(y_star_m.n_cols - 1 - vote_v(i)));
    // double second_max_val = -datum::inf;


    // unsigned int max_ind = 0;
    // double max_val = y_star_m(i, (y_star_m.n_cols - 1) / 2);
    // double second_max_val = max_val;
    // log_prob += R::dnorm(y_star_m(i, (y_star_m.n_cols - 1) / 2), 0, 1.0, true);
    for (unsigned int j = 0; j < half_alpha_size; j++) {
      // double mean =
      mean_v(j) =
        alpha_v_lower(j, q_num) * (beta_val - delta_v_lower(j, q_num));
      // log_prob += R::dnorm(y_star_m(i, j), mean, 1.0, true);

      mean_v(mean_v.n_elem - 1 - j) =
        alpha_v_upper(alpha_v_upper.n_rows - 1 - j, q_num) *
          (beta_val - delta_v_upper(alpha_v_upper.n_rows - 1 - j, q_num));

      // if (j == vote_v(i)) {
      //   continue;
      // }

      if (y_star_m(i, j) > second_max_val &&
          y_star_m(i, j) < max_val) {
        second_max_val = y_star_m(i, j);
      }

      if (y_star_m(i, y_star_m.n_cols - 1 - j) > second_max_val &&
          y_star_m(i, y_star_m.n_cols - 1 - j) < max_val) {
        second_max_val = y_star_m(i, y_star_m.n_cols - 1 - j);
      }

      // if (y_star_m(i, j) > max_val) {
      //   second_max_val = max_val;
      //   max_val = y_star_m(i, j);
      //   max_ind = j;
      // }
      //
      // if (y_star_m(i, y_star_m.n_cols - 1 - j) > max_val) {
      //   second_max_val = max_val;
      //   max_val = y_star_m(i, y_star_m.n_cols - 1 - j);
      //   max_ind = y_star_m.n_cols - 1 - j;
      // }
      //
      // if (y_star_m(i, j) > second_max_val &&
      //       y_star_m(i, j) < max_val) {
      //   second_max_val = y_star_m(i, j);
      // }
      //
      // if (y_star_m(i, y_star_m.n_cols - 1 - j) > second_max_val &&
      //     y_star_m(i, y_star_m.n_cols - 1 - j) < max_val) {
      //   second_max_val = y_star_m(i, y_star_m.n_cols - 1 - j);
      // }
      // log_prob += R::dnorm(y_star_m(i, y_star_m.n_cols - 1 - j),
      //                      mean, 1.0, true);
    }

    // {
    //   double tmp = y_star_m(i, (y_star_m.n_cols - 1) / 2);
    //   if (tmp > max_val) {
    //     second_max_val = max_val;
    //     max_val = tmp;
    //     max_ind = (y_star_m.n_cols - 1) / 2;
    //   }
    //
    //   if (tmp > second_max_val && tmp < max_val) {
    //     second_max_val = tmp;
    //   }
    // }

    // if (max_ind == (y_star_m.n_cols - 1) / 2) {
    if (vote_v(i) == half_alpha_size) {
      log_prob += d_truncnorm(y_star_m(i, (y_star_m.n_cols - 1) / 2), 0, 1.0,
                              second_max_val, datum::inf, 1);
    } else {
      log_prob += d_truncnorm(y_star_m(i, (y_star_m.n_cols - 1) / 2), 0, 1.0,
                              -datum::inf, max_val, 1);
    }
    for (unsigned int j = 0; j < half_alpha_size; j++) {

      if (vote_v(i) == j) {
        double tmp_1 = R::dnorm(y_star_m(i, j), mean_v(j), 1.0, true) +
          d_truncnorm(
            y_star_m(i, y_star_m.n_cols - 1 - j),
            mean_v(mean_v.n_elem - 1 - j),
            1.0, second_max_val, datum::inf, 1);
        double tmp_2 =
          R::dnorm(y_star_m(i, y_star_m.n_cols - 1 - j),
                   mean_v(mean_v.n_elem - 1 - j), 1.0, true) +
          d_truncnorm(
            y_star_m(i, j), mean_v(j),
            1.0, second_max_val, datum::inf, 1);
        if (!is_finite(tmp_1)) {
          log_prob += tmp_2;
        } else if (!is_finite(tmp_2)) {
          log_prob += tmp_1;
        } else {
          log_prob += max(tmp_1, tmp_2);
        }
      } else {
        log_prob += d_truncnorm(
          y_star_m(i, j), mean_v(j), 1.0, -datum::inf, max_val, 1);
        log_prob += d_truncnorm(
          y_star_m(i, (y_star_m.n_cols - 1 - j)),
          mean_v(mean_v.n_elem - 1 - j), 1.0, -datum::inf, max_val, 1);
      }

      // if (!is_finite(log_prob)) {
      //   Rcout << "Infinite" << endl;
      //   Rcout << max_val << endl;
      //   Rcout << second_max_val << endl;
      //   Rcout << y_star_m(i) << endl;
      //   Rcout << vote_v(i) << endl;
      //   Rcout << mean_v << endl;
      // }


      // if (max_ind == j) {
      //   log_prob += d_truncnorm(
      //     y_star_m(i, j), mean_v(j), 1.0, second_max_val, datum::inf, 1);
      // } else {
      //   log_prob += d_truncnorm(
      //     y_star_m(i, j), mean_v(j), 1.0, -datum::inf, max_val, 1);
      // }
      //
      // if (max_ind == (y_star_m.n_cols - 1 - j)) {
      //   log_prob += d_truncnorm(
      //     y_star_m(i, (y_star_m.n_cols - 1 - j)),
      //     mean_v(half_alpha_size - 1 - j), 1.0, second_max_val, datum::inf, 1);
      // } else {
      //   log_prob += d_truncnorm(
      //     y_star_m(i, (y_star_m.n_cols - 1 - j)),
      //     mean_v(half_alpha_size - 1 - j), 1.0, -datum::inf, max_val, 1);
      // }
    }
  }
  return(log_prob);
}

List flip_signs_response_gibbs(
    uvec vote, mat y_star_m,
    vec alpha_v_lower, vec alpha_v_upper, vec beta_v,
    vec delta_v_lower, vec delta_v_upper,
    vec alpha_mean_v, mat alpha_sigma_m,
    vec delta_mean_v, mat delta_sigma_m) {

  uvec orig_vote = vote;
  unsigned int half_alpha_size = alpha_v_lower.n_elem;
  double curr_prob = 0;
  double flip_prob = 0;

  vec flip_alpha_v_lower;
  vec flip_alpha_v_upper;
  vec flip_delta_v_lower;
  vec flip_delta_v_upper;

  {
    vec flip_alpha_v;

    flip_alpha_v =
      sample_alpha_ordinal_independent_lower(alpha_mean_v, alpha_sigma_m);
    flip_alpha_v_lower = flip_alpha_v(span(0, alpha_v_lower.n_elem - 1));
    flip_alpha_v_upper = flip_alpha_v(
      span(alpha_v_lower.n_elem, 2 * alpha_v_lower.n_elem - 1));

    vec flip_delta_v =
      rmvnorm(1, delta_mean_v, delta_sigma_m).t();
    flip_delta_v_lower = flip_delta_v(span(0, delta_v_lower.n_elem - 1));
    flip_delta_v_upper = flip_delta_v(
      span(delta_v_lower.n_elem, 2 * delta_v_lower.n_elem - 1));
  }

  vote = half_alpha_size - vote;
  y_star_m.zeros();
  for (unsigned int j = 0; j < y_star_m.n_rows; j++) {
  // for (unsigned int k = 0; k < half_alpha_size / 2; k++) {
    double max_val = y_star_m.row(j).max();
    if (vote(j) == half_alpha_size) {
      y_star_m(j, half_alpha_size) = max_val + 1;
    } else {
      if (randu() < 0.5) {
        y_star_m(j, vote(j)) = max_val + 1;
      } else {
        y_star_m(j, y_star_m.n_cols - 1 - vote(j)) = max_val + 1;
      }
    }
  }

  for (int i = 0; i < 20; i++) {
    for (int j = 0; j < y_star_m.n_rows; j++) {
      vec output_v = sample_y_star_m(
        y_star_m.row(j).t(), vote(j),
        flip_alpha_v_lower, flip_alpha_v_upper, beta_v(j),
        flip_delta_v_lower, flip_delta_v_upper);
      y_star_m.row(j) = output_v.t();
    }

    // vec match_var_v(num_questions);
    {
      vec out_v = sample_ordinal_probit_matched_alpha_intervals_no_flip(
        y_star_m.cols(join_vert(linspace<uvec>(0, half_alpha_size - 1, half_alpha_size),
                                linspace<uvec>(half_alpha_size + 1, y_star_m.n_cols - 1,
                                               half_alpha_size))),
        beta_v, join_vert(flip_delta_v_lower, flip_delta_v_upper),
        alpha_mean_v, alpha_sigma_m, delta_mean_v, delta_sigma_m);
      flip_alpha_v_lower = out_v(span(0, half_alpha_size - 1));
      flip_alpha_v_upper = out_v(span(half_alpha_size, 2 * half_alpha_size - 1));
      // match_var_v(j) = out_v(out_v.n_elem - 1);
    }

    {
      vec out_v = sample_ordinal_utility_matched_delta(
        y_star_m.cols(0, half_alpha_size - 1),
        y_star_m.cols(half_alpha_size + 1, y_star_m.n_cols - 1),
        flip_alpha_v_lower, flip_alpha_v_upper, beta_v,
        1, delta_mean_v, delta_sigma_m);
      flip_delta_v_lower = out_v(span(0, half_alpha_size - 1));
      flip_delta_v_upper = out_v(span(half_alpha_size, out_v.n_elem - 1));
    }

  }

  for (int i = 0; i < beta_v.n_elem; i++) {
    vec mean_v_lower = alpha_v_lower % (beta_v(i) - delta_v_lower);
    vec mean_v_upper = alpha_v_upper % (beta_v(i) - delta_v_upper);

    curr_prob += log(calc_choice_k_prob(mean_v_lower, mean_v_upper, orig_vote(i)));

    mean_v_lower = flip_alpha_v_lower % (beta_v(i) - flip_delta_v_lower);
    mean_v_upper = flip_alpha_v_upper % (beta_v(i) - flip_delta_v_upper);

    double prob = calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i));
    if (prob < 1e-9) {
      flip_prob += log(1e-9);
    } else {
      flip_prob += log(prob);
    }
    // flip_prob += log(calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i)));
  }

  if (log(randu()) < flip_prob - curr_prob) {
    vec tmp = {1};
    return(List::create(
        join_vert(join_vert(flip_alpha_v_lower, flip_alpha_v_upper,
                            flip_delta_v_lower, flip_delta_v_upper), tmp),
        y_star_m));
  }
  vec tmp = {0};
  return(List::create(join_vert(join_vert(alpha_v_lower, alpha_v_upper,
                             delta_v_lower, delta_v_upper),
                             tmp)));
}

vec convert_alpha_tilde_to_alpha(arma::vec alpha_lower_tilde_v,
                                 arma::vec alpha_upper_tilde_v) {

  unsigned int half_alpha_size = alpha_upper_tilde_v.n_elem;
  vec alpha_lower_v = -exp(alpha_lower_tilde_v);
  vec alpha_upper_v = exp(alpha_upper_tilde_v);
  for (unsigned int i = 1; i < half_alpha_size; i++) {
    alpha_upper_v(i) += alpha_upper_v(i - 1);
    alpha_lower_v(half_alpha_size - 1 - i) +=
      alpha_lower_v(half_alpha_size - 1 - (i - 1));
  }
  return(join_vert(alpha_lower_v, alpha_upper_v));
}

vec convert_alpha_to_alpha_tilde(arma::vec alpha_lower_v,
                                 arma::vec alpha_upper_v) {

  unsigned int half_alpha_size = alpha_lower_v.n_elem;
  vec alpha_lower_tilde_v = -alpha_lower_v;
  vec alpha_upper_tilde_v = alpha_upper_v;
  for (unsigned int i = 0; i < half_alpha_size - 1; i++) {
    alpha_lower_tilde_v(i) -= alpha_lower_tilde_v(i + 1);
    alpha_upper_tilde_v(half_alpha_size - 1 - i) -=
      alpha_upper_tilde_v(half_alpha_size - 1 - (i + 1));
  }
  return(log(join_vert(alpha_lower_tilde_v,
                       alpha_upper_tilde_v)));
}

double obj_fun_rcpp(arma::vec optim_v_ptr,
                    arma::uvec vote, arma::vec beta_v,
                    unsigned int half_alpha_size,
                    vec alpha_mean_v, mat alpha_sigma_m,
                    vec delta_mean_v, mat delta_sigma_m){

  // vec optim_v = *optim_v_ptr;
  vec optim_v = optim_v_ptr;
  vec alpha_v_lower_tilde = optim_v(span(0, half_alpha_size - 1));
  vec alpha_v_upper_tilde = optim_v(span(half_alpha_size, 2 * half_alpha_size - 1));
  vec delta_v_lower = optim_v(span(2 * half_alpha_size, 3 * half_alpha_size - 1));
  vec delta_v_upper = optim_v(span(3 * half_alpha_size, 4 * half_alpha_size - 1));

  vec alpha_v = convert_alpha_tilde_to_alpha(
    alpha_v_lower_tilde, alpha_v_upper_tilde);
  vec alpha_v_lower = alpha_v(span(0, half_alpha_size - 1));
  vec alpha_v_upper = alpha_v(span(half_alpha_size, alpha_v.n_elem - 1));

  double ll =
    as_scalar(
      dmvnorm(join_vert(alpha_v_lower, alpha_v_upper).t(),
              alpha_mean_v, alpha_sigma_m, true)) +
    as_scalar(
      dmvnorm(join_vert(delta_v_lower, delta_v_upper).t(),
              delta_mean_v, delta_sigma_m, true)) +
    accu(alpha_v_lower_tilde) + accu(alpha_v_upper_tilde);
  for (int i = 0; i < beta_v.n_elem; i++) {
    vec mean_v_lower = alpha_v_lower % (beta_v(i) - delta_v_lower);
    vec mean_v_upper = alpha_v_upper % (beta_v(i) - delta_v_upper);

    ll += log(calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i)));
  }
  return(-ll);

//   delta_mean_v_lower <- -4 + (K - 1):1 * -1.5
//   delta_mean_v_upper <- 1:(K - 1) * 1.5 + 4
// #delta_mean_v_lower <- -5 + 4:1 * -5
// #delta_mean_v_upper <- 1:4 * 5
//   delta_sigma_cov = 25 * diag(2 * (K - 1))
//   alpha_sigma_cov = 100 * diag(2 * (K - 1))
//
//   alpha_lower_tilde_v <- optim_v[1:(K - 1)]
//   alpha_upper_tilde_v <- optim_v[K - 1 + 1:(K - 1)]
//   delta_lower_v <- optim_v[2 * (K - 1) + 1:(K - 1)]
//   delta_upper_v <- optim_v[3 * (K - 1) + 1:(K - 1)]
//
//   alpha_lower_v <- -rev(cumsum(rev(exp(alpha_lower_tilde_v))))
//   alpha_upper_v <- cumsum(exp(alpha_upper_tilde_v))
//
//   -(sum(log(ordpum:::calc_waic_ordinal_pum_utility(
//       beta_v, alpha_lower_v, alpha_upper_v,
//       delta_lower_v, delta_lower_v,
//       response_v, K - 1))) +
//         dmvnorm(c(alpha_lower_v, alpha_upper_v),
//                 sigma = alpha_sigma_cov, log = T) +
//                   dmvnorm(c(delta_lower_v, delta_upper_v),
//                           mean = c(delta_mean_v_lower, delta_mean_v_upper),
//                           sigma = delta_sigma_cov, log = T) +
//                             sum(alpha_lower_tilde_v) +
//                             sum(alpha_upper_tilde_v))

}

vec flip_signs_response_optim(
    uvec vote, vec alpha_v_lower, vec alpha_v_upper, vec beta_v,
    vec delta_v_lower, vec delta_v_upper,
    vec alpha_mean_v, mat alpha_sigma_m,
    vec delta_mean_v, mat delta_sigma_m) {

  Rcpp::Environment stats("package:stats");
  Rcpp::Function optim = stats["optim"];

  unsigned int half_alpha_size = alpha_v_lower.n_elem;

  double curr_prob = 0;
  double flip_prob = 0;

  vec flip_alpha_v_lower;
  vec flip_alpha_v_upper;
  vec flip_delta_v_lower;
  vec flip_delta_v_upper;
  double proposal_prob = 0;

  {
    // arma::uvec& vote, arma::vec& beta,
    // double half_alpha_size,
    // vec alpha_mean_v, mat alpha_sigma_m,
    // vec delta_mean_v, mat delta_sigma_m
    vec init_val(4 * alpha_v_lower.n_elem, fill::zeros);
    Rcpp::List opt_results = optim(
      Rcpp::_["par"]    = init_val,
      Rcpp::_["fn"]     = Rcpp::InternalFunction(&obj_fun_rcpp),
      Rcpp::_["method"] = "BFGS", Rcpp::_["hessian"] = true,
      Rcpp::_["vote"] = vote, Rcpp::_["beta"] = beta_v,
      Rcpp::_["half_alpha_size"] = alpha_v_lower.n_elem,
      Rcpp::_["alpha_mean_v"] = alpha_mean_v,
      Rcpp::_["alpha_sigma_m"] = alpha_sigma_m,
      Rcpp::_["delta_mean_v"] = delta_mean_v,
      Rcpp::_["delta_sigma_m"] = delta_sigma_m);

    proposal_prob += as_scalar(dmvt(join_vert(
      convert_alpha_to_alpha_tilde(
        alpha_v_lower, alpha_v_upper),
      delta_v_lower, delta_v_upper),
      Rcpp::as<arma::vec>(opt_results["par"]),
      Rcpp::as<arma::mat>(opt_results["hessian"]), 3, true
    ));
    curr_prob += accu(convert_alpha_to_alpha_tilde(
      alpha_v_lower, alpha_v_upper));

    init_val.zeros();
    Rcpp::List opt_results_flip = optim(
      Rcpp::_["par"]    = init_val,
      Rcpp::_["fn"]     = Rcpp::InternalFunction(&obj_fun_rcpp),
      Rcpp::_["method"] = "BFGS", Rcpp::_["hessian"] = true,
      Rcpp::_["vote"] = half_alpha_size - vote, Rcpp::_["beta"] = beta_v,
      Rcpp::_["half_alpha_size"] = half_alpha_size,
      Rcpp::_["alpha_mean_v"] = alpha_mean_v,
      Rcpp::_["alpha_sigma_m"] = alpha_sigma_m,
      Rcpp::_["delta_mean_v"] = delta_mean_v,
      Rcpp::_["delta_sigma_m"] = delta_sigma_m);

    vec flip_v = rmvt(1, Rcpp::as<arma::vec>(opt_results_flip["par"]),
                      Rcpp::as<arma::mat>(opt_results_flip["hessian"]), 3).row(0).t();

    // mat tmp(4 * alpha_v_lower.n_elem, 4 * alpha_v_lower.n_elem, fill::eye);
    // vec init_val(4 * alpha_v_lower.n_elem, fill::zeros);
    // vec flip_v = rmvt(1, init_val, tmp, 3).row(0).t();
    proposal_prob -= as_scalar(
      dmvt(flip_v, Rcpp::as<arma::vec>(opt_results_flip["par"]),
           Rcpp::as<arma::mat>(opt_results_flip["hessian"]), 3));
    flip_prob += accu(flip_v(span(0, 2 * half_alpha_size - 1)));

    vec flip_alpha_v =
      convert_alpha_tilde_to_alpha(
        flip_v(span(0, half_alpha_size - 1)),
        flip_v(span(half_alpha_size, 2 * half_alpha_size - 1)));
    flip_alpha_v_lower = flip_alpha_v(span(0, half_alpha_size - 1));
    flip_alpha_v_upper = flip_alpha_v(
      span(half_alpha_size, 2 * half_alpha_size - 1));

    flip_delta_v_lower = flip_v(span(
      2 * half_alpha_size, 3 * half_alpha_size - 1));
    flip_delta_v_upper = flip_v(
      span(3 * half_alpha_size, 4 * half_alpha_size - 1));
  }

  curr_prob += as_scalar(
      dmvnorm(join_vert(alpha_v_lower, alpha_v_upper).t(),
                        alpha_mean_v, alpha_sigma_m, true)) +
    as_scalar(
      dmvnorm(join_vert(delta_v_lower, delta_v_upper).t(),
                        delta_mean_v, delta_sigma_m, true));

  flip_prob += as_scalar(
    dmvnorm(join_vert(flip_alpha_v_lower, flip_alpha_v_upper).t(),
            alpha_mean_v, alpha_sigma_m, true)) +
              as_scalar(
                dmvnorm(join_vert(flip_delta_v_lower, flip_delta_v_upper).t(),
                        delta_mean_v, delta_sigma_m, true));

  for (int i = 0; i < beta_v.n_elem; i++) {
    vec mean_v_lower = alpha_v_lower % (beta_v(i) - delta_v_lower);
    vec mean_v_upper = alpha_v_upper % (beta_v(i) - delta_v_upper);

    curr_prob += log(calc_choice_k_prob(mean_v_lower, mean_v_upper, vote(i)));

    mean_v_lower = flip_alpha_v_lower % (beta_v(i) - flip_delta_v_lower);
    mean_v_upper = flip_alpha_v_upper % (beta_v(i) - flip_delta_v_upper);

    double prob = calc_choice_k_prob(mean_v_lower, mean_v_upper, alpha_v_lower.n_elem - vote(i));
    if (prob < 1e-9) {
      flip_prob += log(1e-9);
    } else {
      flip_prob += log(prob);
    }
    // flip_prob += log(calc_choice_k_prob(mean_v_lower, mean_v_upper, alpha_v_lower.n_elem - vote(i)));
  }

  if (log(randu()) < (flip_prob - curr_prob + proposal_prob)) {
    vec tmp = {1};
    return(join_vert(join_vert(flip_alpha_v_lower, flip_alpha_v_upper,
                               flip_delta_v_lower, flip_delta_v_upper), tmp));
  }
  vec tmp = {0};
  return(join_vert(join_vert(alpha_v_lower, alpha_v_upper,
                             delta_v_lower, delta_v_upper),
                             tmp));
}

// vec flip_current_param_val_v_gen_responses(vec current_param_val_v,
//                                                    unsigned int num_questions,
//                                                    unsigned int max_response_num) {
//
//   current_param_val_v(span(0, alpha_v_lower_start_ind - 1)) =
//     -current_param_val_v(span(0, alpha_v_lower_start_ind - 1));
//   current_param_val_v(span(delta_v_lower_start_ind, current_param_val_v.n_elem - 1)) =
//     -current_param_val_v(span(delta_v_lower_start_ind, current_param_val_v.n_elem - 1));
//     for (unsigned int j = 0; j < num_questions; j++) {
//       for (unsigned int k = 0; k < max_response_num; k++) {
//         double tmp = current_param_val_v(alpha_v_lower_start_ind + j * max_response_num + k);
//         current_param_val_v(alpha_v_lower_start_ind + j * max_response_num + k) =
//           -current_param_val_v(alpha_v_upper_start_ind + (j + 1) * max_response_num - 1 - k);
//         current_param_val_v(alpha_v_upper_start_ind + (j + 1) * max_response_num - 1 - k) =
//           -tmp;
//       }
//     }
//   return(current_param_val_v)
// }

// [[Rcpp::export]]
List sample_ordinal_utility_probit(
  arma::uvec vote_v, arma::uvec respondent_v, arma::uvec question_v,
  arma::mat all_param_draws, arma::mat y_star_m,
  int leg_start_ind, int alpha_v_lower_start_ind, int alpha_v_upper_start_ind,
  int delta_v_lower_start_ind, int delta_v_upper_start_ind,
  double leg_mean, double leg_sd, arma::vec alpha_mean_v, arma::mat alpha_cov_s,
  arma::vec delta_mean_v, arma::mat delta_cov_s,
  int num_iter, int start_iter, int keep_iter, int pos_ind, int neg_ind) {

  vec current_param_val_v = all_param_draws.row(0).t();
  int half_alpha_size = (y_star_m.n_cols - 1) / 2;
  int num_questions = (alpha_v_upper_start_ind - alpha_v_lower_start_ind) / half_alpha_size;
  int num_ind = alpha_v_lower_start_ind - leg_start_ind;

  for (int i = 0; i < num_iter; i++) {
    if (i % 100 == 0) {
      Rcout << i << "\n";
    }

    for (int j = 0; j < y_star_m.n_rows; j++) {
      uvec interested_q_inds =
        linspace<uvec>(question_v(j) * half_alpha_size,
                       (question_v(j) + 1) * half_alpha_size - 1, half_alpha_size);
      vec output_v = sample_y_star_m(
        y_star_m.row(j).t(), vote_v(j),
        current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
        current_param_val_v(alpha_v_upper_start_ind + interested_q_inds),
        current_param_val_v(leg_start_ind + respondent_v(j)),
        current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
        current_param_val_v(delta_v_upper_start_ind + interested_q_inds));
      y_star_m.row(j) = output_v.t();
    }

    for (unsigned int j = 0; j < num_ind; j++) {
      uvec interested_inds = find(respondent_v == j);
      uvec interested_q_dim_inds(interested_inds.n_elem * half_alpha_size);
      for (int k = 0; k < interested_inds.n_elem; k++) {
        int question_ind = question_v(interested_inds(k));

        interested_q_dim_inds(
          span(k * half_alpha_size,
               (k + 1) * half_alpha_size - 1)) =
          linspace<uvec>(question_ind * half_alpha_size,
                         (question_ind + 1) * half_alpha_size - 1, half_alpha_size);
      }

      current_param_val_v(leg_start_ind + j) =
        sample_ordinal_utility_probit_beta(
          y_star_m.submat(interested_inds,
            linspace<uvec>(0, half_alpha_size - 1, half_alpha_size)),
          y_star_m.submat(interested_inds,
            linspace<uvec>(half_alpha_size + 1, y_star_m.n_cols - 1, half_alpha_size)),
          reshape(current_param_val_v(alpha_v_lower_start_ind + interested_q_dim_inds),
                  half_alpha_size, interested_inds.n_elem).t(),
          reshape(current_param_val_v(alpha_v_upper_start_ind + interested_q_dim_inds),
                  half_alpha_size, interested_inds.n_elem).t(),
          reshape(current_param_val_v(delta_v_lower_start_ind + interested_q_dim_inds),
                  half_alpha_size, interested_inds.n_elem).t(),
          reshape(current_param_val_v(delta_v_upper_start_ind + interested_q_dim_inds),
                  half_alpha_size, interested_inds.n_elem).t(),
          leg_mean, leg_sd);
    }

    vec match_var_v(num_questions);
    for (unsigned int j = 0; j < num_questions; j++) {

      uvec interested_inds = find(question_v == j);
      uvec interested_q_inds =
        linspace<uvec>(j * half_alpha_size,
                       (j + 1) * half_alpha_size - 1, half_alpha_size);
      vec out_v = sample_ordinal_probit_matched_alpha_intervals(
          y_star_m.submat(interested_inds,
                          join_vert(linspace<uvec>(0, half_alpha_size - 1, half_alpha_size),
                                    linspace<uvec>(half_alpha_size + 1, y_star_m.n_cols - 1, half_alpha_size))),
          current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
          join_vert(current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
                    current_param_val_v(delta_v_upper_start_ind + interested_q_inds)),
          alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s);
      current_param_val_v(alpha_v_lower_start_ind + interested_q_inds) =
        out_v(span(0, half_alpha_size - 1));
      current_param_val_v(alpha_v_upper_start_ind + interested_q_inds) =
        out_v(span(half_alpha_size, 2 * half_alpha_size - 1));
      match_var_v(j) = out_v(out_v.n_elem - 1);
    }

    for (unsigned int j = 0; j < num_questions; j++) {
      uvec interested_inds = find(question_v == j);
      uvec interested_q_inds =
        linspace<uvec>(j * half_alpha_size,
                       (j + 1) * half_alpha_size - 1, half_alpha_size);

      vec out_v = sample_ordinal_utility_matched_delta(
        y_star_m.submat(interested_inds,
          linspace<uvec>(0, half_alpha_size - 1, half_alpha_size)),
        y_star_m.submat(interested_inds,
          linspace<uvec>(half_alpha_size + 1, y_star_m.n_cols - 1, half_alpha_size)),
        current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
        current_param_val_v(alpha_v_upper_start_ind + interested_q_inds),
        current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
        match_var_v(j), delta_mean_v, delta_cov_s);
      current_param_val_v(delta_v_lower_start_ind + interested_q_inds) =
        out_v(span(0, half_alpha_size - 1));
      current_param_val_v(delta_v_upper_start_ind + interested_q_inds) =
        out_v(span(half_alpha_size, out_v.n_elem - 1));
    }

    if (pos_ind > -1 && (current_param_val_v(leg_start_ind + pos_ind) < 0)) {
      current_param_val_v = -current_param_val_v;
    }

    if (neg_ind > -1 && pos_ind < 0 && (current_param_val_v(leg_start_ind + neg_ind) > 0)) {
      current_param_val_v = -current_param_val_v;
    }

    int post_burn_i = i - start_iter + 1;
    if (i >= start_iter && (fmod(post_burn_i, keep_iter) == 0)) {
      int keep_iter_ind = post_burn_i / keep_iter - 1;
      all_param_draws.row(keep_iter_ind) = current_param_val_v.t();
    }
  }

  return(List::create(Named("param_draws") = all_param_draws,
                      Named("y_star_m") = y_star_m));
}

// [[Rcpp::export]]
List sample_ordinal_utility_probit_gen_choices(
    arma::uvec vote_v, arma::uvec respondent_v, arma::uvec question_v,
    arma::uvec question_num_choices_m1_v,
    arma::mat all_param_draws, arma::mat y_star_m,
    int leg_start_ind, int alpha_v_lower_start_ind, int alpha_v_upper_start_ind,
    int delta_v_lower_start_ind, int delta_v_upper_start_ind,
    double leg_mean, double leg_sd, arma::vec alpha_mean_v, arma::mat alpha_cov_s,
    arma::vec delta_mean_v, arma::mat delta_cov_s,
    int num_iter, int start_iter, int keep_iter, int pos_ind, int neg_ind) {

  vec current_param_val_v = all_param_draws.row(0).t();
  int num_questions = question_num_choices_m1_v.n_elem;
  int num_ind = alpha_v_lower_start_ind - leg_start_ind;
  int max_response_num = (y_star_m.n_cols - 1) / 2;
  uvec num_responses(num_questions + 1);
  num_responses(span(1, num_questions)) = cumsum(question_num_choices_m1_v);
  num_responses(0) = 0;
  vec y_star_log_ll(all_param_draws.n_rows);

  for (int i = 0; i < num_iter; i++) {
    if (i % 100 == 0) {
      Rcout << i << "\n";
    }

    for (unsigned int j = 0; j < y_star_m.n_rows; j++) {

      int half_alpha_size = question_num_choices_m1_v(question_v(j));

      uvec interested_q_inds =
        linspace<uvec>(question_v(j) * max_response_num,
                       question_v(j) * max_response_num + half_alpha_size - 1,
                       half_alpha_size);
      uvec tmp = {(y_star_m.n_cols - 1) / 2};
      uvec interested_row_ind = {j};
      vec output_v = sample_y_star_m(
        y_star_m.submat(interested_row_ind,
                        join_vert(
                          linspace<uvec>(0, half_alpha_size - 1, half_alpha_size), tmp,
                          linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                         half_alpha_size))).t(),
        vote_v(j),
        current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
        current_param_val_v(
          alpha_v_upper_start_ind + interested_q_inds -
            half_alpha_size + max_response_num),
        current_param_val_v(leg_start_ind + respondent_v(j)),
        current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
        current_param_val_v(delta_v_upper_start_ind + interested_q_inds -
          half_alpha_size + max_response_num));

      y_star_m(j, span(0, half_alpha_size - 1)) = output_v(span(0, half_alpha_size - 1)).t();
      y_star_m(j, span(y_star_m.n_cols - half_alpha_size,
                       y_star_m.n_cols - 1)) =
        output_v(span(half_alpha_size + 1, 2 * half_alpha_size)).t();
      y_star_m(j, max_response_num) = output_v(half_alpha_size);
    }

    for (unsigned int j = 0; j < num_ind; j++) {
      uvec interested_inds = find(respondent_v == j);
      uvec interested_q_dim_inds(interested_inds.n_elem * max_response_num);
      for (int k = 0; k < interested_inds.n_elem; k++) {
        int question_ind = question_v(interested_inds(k));

        interested_q_dim_inds(
          span(k * max_response_num,
               (k + 1) * max_response_num - 1)) =
                 linspace<uvec>(question_ind * max_response_num,
                                (question_ind + 1) * max_response_num - 1, max_response_num);
      }

      current_param_val_v(leg_start_ind + j) =
        sample_ordinal_utility_probit_beta(
          y_star_m.submat(interested_inds,
                          linspace<uvec>(0, max_response_num - 1, max_response_num)),
          y_star_m.submat(interested_inds,
                          linspace<uvec>(max_response_num + 1, y_star_m.n_cols - 1,
                                         max_response_num)),
          reshape(current_param_val_v(alpha_v_lower_start_ind + interested_q_dim_inds),
                  max_response_num, interested_inds.n_elem).t(),
          reshape(current_param_val_v(alpha_v_upper_start_ind + interested_q_dim_inds),
                  max_response_num, interested_inds.n_elem).t(),
          reshape(current_param_val_v(delta_v_lower_start_ind + interested_q_dim_inds),
                  max_response_num, interested_inds.n_elem).t(),
          reshape(current_param_val_v(delta_v_upper_start_ind + interested_q_dim_inds),
                  max_response_num, interested_inds.n_elem).t(),
          leg_mean, leg_sd);
    }

    vec match_var_v(num_questions);
    for (unsigned int j = 0; j < num_questions; j++) {

      int half_alpha_size = question_num_choices_m1_v(j);

      uvec interested_inds = find(question_v == j);
      uvec interested_q_inds =
        linspace<uvec>(j * max_response_num,
                       j * max_response_num + half_alpha_size - 1, half_alpha_size);
      uvec interested_q_inds_upper =
        linspace<uvec>((j + 1) * max_response_num - half_alpha_size,
                       (j + 1) * max_response_num - 1, half_alpha_size);

      // Rcout << half_alpha_size << endl;
      // Rcout << interested_q_inds << endl;

      vec out_v = sample_ordinal_probit_matched_alpha_intervals_no_flip(
        y_star_m.submat(interested_inds,
                        join_vert(linspace<uvec>(0, half_alpha_size - 1, half_alpha_size),
                                  linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                                 half_alpha_size))),
                        current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
                        join_vert(current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
                                  current_param_val_v(delta_v_upper_start_ind + interested_q_inds_upper)),
                        alpha_mean_v(span(max_response_num - half_alpha_size,
                                          max_response_num + half_alpha_size - 1)),
                        alpha_cov_s(span(max_response_num - half_alpha_size,
                                         max_response_num + half_alpha_size - 1),
                                    span(max_response_num - half_alpha_size,
                                         max_response_num + half_alpha_size - 1)),
                        delta_mean_v, delta_cov_s);

      // Rcout << out_v.n_elem << endl;
      current_param_val_v(alpha_v_lower_start_ind + interested_q_inds) =
        out_v(span(0, half_alpha_size - 1));
      current_param_val_v(
        alpha_v_upper_start_ind + interested_q_inds_upper) =
          out_v(span(half_alpha_size, 2 * half_alpha_size - 1));
      match_var_v(j) = out_v(out_v.n_elem - 1);
    }

    for (unsigned int j = 0; j < num_questions; j++) {
      uvec interested_inds = find(question_v == j);
      int half_alpha_size = question_num_choices_m1_v(j);

      uvec interested_q_inds =
        linspace<uvec>(j * max_response_num,
                       j * max_response_num + half_alpha_size - 1, half_alpha_size);
      uvec interested_q_inds_upper =
        linspace<uvec>((j + 1) * max_response_num - half_alpha_size,
                       (j + 1) * max_response_num - 1, half_alpha_size);

      vec out_v = sample_ordinal_utility_matched_delta(
        y_star_m.submat(interested_inds,
                        linspace<uvec>(0, half_alpha_size - 1, half_alpha_size)),
        y_star_m.submat(interested_inds,
                        linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                       half_alpha_size)),
        current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
        current_param_val_v(alpha_v_upper_start_ind + interested_q_inds_upper),
        current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
        match_var_v(j),
        delta_mean_v(span(max_response_num - half_alpha_size,
                          max_response_num + half_alpha_size - 1)),
        delta_cov_s(span(max_response_num - half_alpha_size,
                         max_response_num + half_alpha_size - 1),
                    span(max_response_num - half_alpha_size,
                         max_response_num + half_alpha_size - 1)));

      // Rcout << out_v.n_elem << endl;

      current_param_val_v(delta_v_lower_start_ind + interested_q_inds) =
        out_v(span(0, half_alpha_size - 1));
      current_param_val_v(
        delta_v_upper_start_ind + interested_q_inds_upper) =
        out_v(span(half_alpha_size, out_v.n_elem - 1));
    }

    if (pos_ind > -1 && (current_param_val_v(leg_start_ind + pos_ind) < 0)) {
      current_param_val_v(span(0, alpha_v_lower_start_ind - 1)) =
        -current_param_val_v(span(0, alpha_v_lower_start_ind - 1));
        current_param_val_v(span(delta_v_lower_start_ind, current_param_val_v.n_elem - 1)) =
        -current_param_val_v(span(delta_v_lower_start_ind, current_param_val_v.n_elem - 1));
        for (unsigned int j = 0; j < num_questions; j++) {
          for (unsigned int k = 0; k < max_response_num; k++) {
            double tmp = current_param_val_v(alpha_v_lower_start_ind + j * max_response_num + k);
            current_param_val_v(alpha_v_lower_start_ind + j * max_response_num + k) =
              -current_param_val_v(alpha_v_upper_start_ind + (j + 1) * max_response_num - 1 - k);
              current_param_val_v(alpha_v_upper_start_ind + (j + 1) * max_response_num - 1 - k) =
              -tmp;
          }
        }
    }

    if (neg_ind > -1 && pos_ind < 0 && (current_param_val_v(leg_start_ind + neg_ind) > 0)) {
      current_param_val_v(span(0, alpha_v_lower_start_ind - 1)) =
        -current_param_val_v(span(0, alpha_v_lower_start_ind - 1));
        current_param_val_v(span(delta_v_lower_start_ind, current_param_val_v.n_elem - 1)) =
        -current_param_val_v(span(delta_v_lower_start_ind, current_param_val_v.n_elem - 1));
        for (unsigned int j = 0; j < num_questions; j++) {
          for (unsigned int k = 0; k < max_response_num; k++) {
            double tmp = current_param_val_v(alpha_v_lower_start_ind + j * max_response_num + k);
            current_param_val_v(alpha_v_lower_start_ind + j * max_response_num + k) =
              -current_param_val_v(alpha_v_upper_start_ind + (j + 1) * max_response_num - 1 - k);
              current_param_val_v(alpha_v_upper_start_ind + (j + 1) * max_response_num - 1 - k) =
              -tmp;
          }
        }
    }

    int post_burn_i = i - start_iter + 1;
    if (i >= start_iter && (fmod(post_burn_i, keep_iter) == 0)) {
      int keep_iter_ind = post_burn_i / keep_iter - 1;
      all_param_draws.row(keep_iter_ind) = current_param_val_v.t();
      y_star_log_ll(keep_iter_ind) =
        calc_log_ll_y_star(
          vote_v,
          y_star_m, reshape(current_param_val_v(span(
              alpha_v_lower_start_ind, alpha_v_lower_start_ind +
                max_response_num * num_questions - 1)),
                max_response_num, num_questions),
                reshape(current_param_val_v(span(
                    alpha_v_upper_start_ind, alpha_v_upper_start_ind +
                      max_response_num * num_questions - 1)),
                      max_response_num, num_questions),
                      current_param_val_v(span(leg_start_ind, leg_start_ind + num_ind - 1)),
                      reshape(current_param_val_v(span(
                          delta_v_lower_start_ind, delta_v_lower_start_ind +
                            max_response_num * num_questions - 1)),
                            max_response_num, num_questions),
                            reshape(current_param_val_v(span(
                                delta_v_upper_start_ind, delta_v_upper_start_ind +
                                  max_response_num * num_questions - 1)),
                                  max_response_num, num_questions),
                                  respondent_v, question_v, question_num_choices_m1_v);
    }
  }

  return(List::create(Named("param_draws") = all_param_draws,
                      Named("y_star_m") = y_star_m,
                      Named("y_star_log_ll") = y_star_log_ll));
}

// [[Rcpp::export]]
List sample_ordinal_utility_probit_gen_choices_flip_responses(
    arma::uvec vote_v, arma::uvec respondent_v, arma::uvec question_v,
    arma::uvec question_num_choices_m1_v,
    arma::mat all_param_draws, arma::mat y_star_m,
    int leg_start_ind, int alpha_v_lower_start_ind, int alpha_v_upper_start_ind,
    int delta_v_lower_start_ind, int delta_v_upper_start_ind,
    double leg_mean, double leg_sd, arma::vec alpha_mean_v, arma::mat alpha_cov_s,
    arma::vec delta_mean_v, arma::mat delta_cov_s,
    int num_iter, int start_iter, int keep_iter, int pos_ind, int neg_ind) {

  vec current_param_val_v = all_param_draws.row(0).t();
  int num_questions = question_num_choices_m1_v.n_elem;
  int num_ind = alpha_v_lower_start_ind - leg_start_ind;
  int max_response_num = (y_star_m.n_cols - 1) / 2;
  uvec num_responses(num_questions + 1);
  num_responses(span(1, num_questions)) = cumsum(question_num_choices_m1_v);
  num_responses(0) = 0;
  mat swap_m(num_iter / 50, num_questions, fill::zeros);

  for (int i = 0; i < num_iter; i++) {
    if (i % 100 == 0) {
      Rcout << i << "\n";
    }

    for (unsigned int j = 0; j < y_star_m.n_rows; j++) {

      int half_alpha_size = question_num_choices_m1_v(question_v(j));

      uvec interested_q_inds =
        linspace<uvec>(question_v(j) * max_response_num,
                       question_v(j) * max_response_num + half_alpha_size - 1,
                       half_alpha_size);
      uvec tmp = {(y_star_m.n_cols - 1) / 2};
      uvec interested_row_ind = {j};
      vec output_v = sample_y_star_m(
        y_star_m.submat(interested_row_ind,
                        join_vert(
                          linspace<uvec>(0, half_alpha_size - 1, half_alpha_size), tmp,
                          linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                         half_alpha_size))).t(),
                                         vote_v(j),
                                         current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
                                         current_param_val_v(
                                           alpha_v_upper_start_ind + interested_q_inds -
                                             half_alpha_size + max_response_num),
                                             current_param_val_v(leg_start_ind + respondent_v(j)),
                                             current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
                                             current_param_val_v(delta_v_upper_start_ind + interested_q_inds -
                                               half_alpha_size + max_response_num));

      y_star_m(j, span(0, half_alpha_size - 1)) = output_v(span(0, half_alpha_size - 1)).t();
      y_star_m(j, span(y_star_m.n_cols - half_alpha_size,
                       y_star_m.n_cols - 1)) =
                         output_v(span(half_alpha_size + 1, 2 * half_alpha_size)).t();
      y_star_m(j, max_response_num) = output_v(half_alpha_size);
    }

    for (unsigned int j = 0; j < num_ind; j++) {
      uvec interested_inds = find(respondent_v == j);
      uvec interested_q_dim_inds(interested_inds.n_elem * max_response_num);
      for (int k = 0; k < interested_inds.n_elem; k++) {
        int question_ind = question_v(interested_inds(k));

        interested_q_dim_inds(
          span(k * max_response_num,
               (k + 1) * max_response_num - 1)) =
                 linspace<uvec>(question_ind * max_response_num,
                                (question_ind + 1) * max_response_num - 1, max_response_num);
      }

      current_param_val_v(leg_start_ind + j) =
        sample_ordinal_utility_probit_beta(
          y_star_m.submat(interested_inds,
                          linspace<uvec>(0, max_response_num - 1, max_response_num)),
                          y_star_m.submat(interested_inds,
                                          linspace<uvec>(max_response_num + 1, y_star_m.n_cols - 1,
                                                         max_response_num)),
                                                         reshape(current_param_val_v(alpha_v_lower_start_ind + interested_q_dim_inds),
                                                                 max_response_num, interested_inds.n_elem).t(),
                                                                 reshape(current_param_val_v(alpha_v_upper_start_ind + interested_q_dim_inds),
                                                                         max_response_num, interested_inds.n_elem).t(),
                                                                         reshape(current_param_val_v(delta_v_lower_start_ind + interested_q_dim_inds),
                                                                                 max_response_num, interested_inds.n_elem).t(),
                                                                                 reshape(current_param_val_v(delta_v_upper_start_ind + interested_q_dim_inds),
                                                                                         max_response_num, interested_inds.n_elem).t(),
                                                                                         leg_mean, leg_sd);
    }

    vec match_var_v(num_questions);
    for (unsigned int j = 0; j < num_questions; j++) {

      int half_alpha_size = question_num_choices_m1_v(j);

      uvec interested_inds = find(question_v == j);
      uvec interested_q_inds =
        linspace<uvec>(j * max_response_num,
                       j * max_response_num + half_alpha_size - 1, half_alpha_size);
      uvec interested_q_inds_upper =
        linspace<uvec>((j + 1) * max_response_num - half_alpha_size,
                       (j + 1) * max_response_num - 1, half_alpha_size);

      vec out_v = sample_ordinal_probit_matched_alpha_intervals_no_flip(
        y_star_m.submat(interested_inds,
                        join_vert(linspace<uvec>(0, half_alpha_size - 1, half_alpha_size),
                                  linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                                 half_alpha_size))),
                                                 current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
                                                 join_vert(current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
                                                           current_param_val_v(delta_v_upper_start_ind + interested_q_inds_upper)),
                                                           alpha_mean_v(span(max_response_num - half_alpha_size,
                                                                             max_response_num + half_alpha_size - 1)),
                                                                             alpha_cov_s(span(max_response_num - half_alpha_size,
                                                                                              max_response_num + half_alpha_size - 1),
                                                                                              span(max_response_num - half_alpha_size,
                                                                                                   max_response_num + half_alpha_size - 1)),
                                                                                                   delta_mean_v, delta_cov_s);
      current_param_val_v(alpha_v_lower_start_ind + interested_q_inds) =
        out_v(span(0, half_alpha_size - 1));
      current_param_val_v(
        alpha_v_upper_start_ind + interested_q_inds_upper) =
          out_v(span(half_alpha_size, 2 * half_alpha_size - 1));
      match_var_v(j) = out_v(out_v.n_elem - 1);
    }

    for (unsigned int j = 0; j < num_questions; j++) {
      uvec interested_inds = find(question_v == j);
      int half_alpha_size = question_num_choices_m1_v(j);

      uvec interested_q_inds =
        linspace<uvec>(j * max_response_num,
                       j * max_response_num + half_alpha_size - 1, half_alpha_size);
      uvec interested_q_inds_upper =
        linspace<uvec>((j + 1) * max_response_num - half_alpha_size,
                       (j + 1) * max_response_num - 1, half_alpha_size);

      vec out_v = sample_ordinal_utility_matched_delta(
        y_star_m.submat(interested_inds,
                        linspace<uvec>(0, half_alpha_size - 1, half_alpha_size)),
                        y_star_m.submat(interested_inds,
                                        linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                                       half_alpha_size)),
                                                       current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
                                                       current_param_val_v(alpha_v_upper_start_ind + interested_q_inds_upper),
                                                       current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
                                                       match_var_v(j),
                                                       delta_mean_v(span(max_response_num - half_alpha_size,
                                                                         max_response_num + half_alpha_size - 1)),
                                                                         delta_cov_s(span(max_response_num - half_alpha_size,
                                                                                          max_response_num + half_alpha_size - 1),
                                                                                          span(max_response_num - half_alpha_size,
                                                                                               max_response_num + half_alpha_size - 1)));
      current_param_val_v(delta_v_lower_start_ind + interested_q_inds) =
        out_v(span(0, half_alpha_size - 1));
      current_param_val_v(
        delta_v_upper_start_ind + interested_q_inds_upper) =
          out_v(span(half_alpha_size, out_v.n_elem - 1));
    }

    if (i > 0 && ((i + 1) % 50 == 0)) {
      for (unsigned int j = 0; j < num_questions; j++) {
        uvec interested_inds = find(question_v == j);
        int half_alpha_size = question_num_choices_m1_v(j);

        uvec interested_q_inds =
          linspace<uvec>(j * max_response_num,
                         j * max_response_num + half_alpha_size - 1,
                         half_alpha_size);

        uvec interested_q_inds_upper =
          linspace<uvec>((j + 1) * max_response_num - half_alpha_size,
                         (j + 1) * max_response_num - 1, half_alpha_size);
        uvec middle_ind = {static_cast<unsigned int>(max_response_num)};

        // vec out_v = flip_signs_response_gibbs(
        // flip_signs_response_optim(
        //   uvec vote, vec alpha_v_lower, vec alpha_v_upper, vec beta_v,
        //   vec delta_v_lower, vec delta_v_upper,
        //   vec alpha_mean_v, mat alpha_sigma_m,
        //   vec delta_mean_v, mat delta_sigma_m)
        vec out_v = flip_signs_response_optim(
          vote_v(interested_inds),
          // y_star_m.submat(
          //   interested_inds,
          //   join_vert(linspace<uvec>(0, half_alpha_size - 1, half_alpha_size),
          //             middle_ind,
          //             linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
          //                            half_alpha_size))),
          current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
          current_param_val_v(alpha_v_upper_start_ind + interested_q_inds_upper),
          current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
          current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
          current_param_val_v(delta_v_upper_start_ind + interested_q_inds_upper),
          alpha_mean_v(span(max_response_num - half_alpha_size,
                            max_response_num + half_alpha_size - 1)),
          alpha_cov_s(span(max_response_num - half_alpha_size,
                           max_response_num + half_alpha_size - 1),
                      span(max_response_num - half_alpha_size,
                           max_response_num + half_alpha_size - 1)),
          delta_mean_v(span(max_response_num - half_alpha_size,
                            max_response_num + half_alpha_size - 1)),
          delta_cov_s(span(max_response_num - half_alpha_size,
                           max_response_num + half_alpha_size - 1),
                      span(max_response_num - half_alpha_size,
                           max_response_num + half_alpha_size - 1)));
        // vec out_v = out_v_info[0];
        current_param_val_v(alpha_v_lower_start_ind + interested_q_inds) =
          out_v(span(0, half_alpha_size - 1));
        current_param_val_v(alpha_v_upper_start_ind + interested_q_inds_upper) =
          out_v(span(half_alpha_size, 2 * half_alpha_size - 1));
        current_param_val_v(delta_v_lower_start_ind + interested_q_inds) =
          out_v(span(2 * half_alpha_size, 3 * half_alpha_size - 1));
        current_param_val_v(delta_v_upper_start_ind + interested_q_inds_upper) =
          out_v(span(3 * half_alpha_size, 4 * half_alpha_size - 1));
        if (out_v(out_v.n_elem - 1) > 0.5) {
          vote_v(interested_inds) = half_alpha_size - vote_v(interested_inds);
          // for (unsigned int k = 0; k < half_alpha_size / 2; k++) {
          //   uvec low_swap_ind = {k};
          //   uvec up_swap_ind = {half_alpha_size - 1 - k};
          //   vec tmp =
          //     y_star_m.submat(interested_inds, low_swap_ind);
          //   y_star_m.submat(interested_inds, low_swap_ind) =
          //     y_star_m.submat(interested_inds, up_swap_ind);
          //   y_star_m.submat(interested_inds, up_swap_ind) = tmp;
          //
          //   tmp = y_star_m.submat(interested_inds, low_swap_ind +
          //     y_star_m.n_cols - half_alpha_size);
          //   y_star_m.submat(interested_inds, low_swap_ind +
          //     y_star_m.n_cols - half_alpha_size) =
          //       y_star_m.submat(interested_inds, up_swap_ind +
          //         y_star_m.n_cols - half_alpha_size);
          //   y_star_m.submat(interested_inds, up_swap_ind +
          //     y_star_m.n_cols - half_alpha_size) = tmp;
          // }
          // mat tmp = out_v_info[1];
          // y_star_m.submat(
          //   interested_inds,
          //   join_vert(linspace<uvec>(0, half_alpha_size - 1, half_alpha_size),
          //             middle_ind,
          //             linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
          //                            half_alpha_size))) =
          //   tmp;
          y_star_m(interested_inds).zeros();
          for (unsigned int j = 0; j < interested_inds.n_elem; j++) {
            // for (unsigned int k = 0; k < half_alpha_size / 2; k++) {
            // double max_val = y_star_m.row(j).max();
            if (vote_v(interested_inds(j)) == half_alpha_size) {
              y_star_m(interested_inds(j), max_response_num) = 1;
            } else {
              if (randu() < 0.5) {
                y_star_m(interested_inds(j),  vote_v(interested_inds(j))) = 1;
              } else {
                y_star_m(interested_inds(j), y_star_m.n_cols - 1 -  vote_v(interested_inds(j))) = 1;
              }
            }
          }
          swap_m((i + 1) / 50 - 1, j) = 1;

          // y_star_m.submat(interested_inds,
          //                 linspace<uvec>(0, half_alpha_size - 1,
          //                                half_alpha_size));
          // y_star_m.submat(interested_inds,
          //                 linspace<uvec>(y_star_m.n_cols - half_alpha_size,
          //                                y_star_m.n_cols - 1, half_alpha_size));
          //                 join_vert(linspace<uvec>(0, half_alpha_size - 1, half_alpha_size),
          //                           linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
          //                                          half_alpha_size)))
        }
        // vote_v(interested_inds) = out_v(
        //   span(4 * half_alpha_size, out_v.n_elem - 1));
      }
    }


    int post_burn_i = i - start_iter + 1;
    if (i >= start_iter && (fmod(post_burn_i, keep_iter) == 0)) {
      int keep_iter_ind = post_burn_i / keep_iter - 1;
      all_param_draws.row(keep_iter_ind) = current_param_val_v.t();
    }
  }

  return(List::create(Named("param_draws") = all_param_draws,
                      Named("y_star_m") = y_star_m,
                      Named("swap_tracker") = swap_m));
}


// [[Rcpp::export]]
List sample_ordinal_utility_probit_gen_choices_flip_responses_prior(
    arma::uvec vote_v, arma::uvec respondent_v, arma::uvec question_v,
    arma::uvec question_num_choices_m1_v,
    arma::mat all_param_draws, arma::mat y_star_m,
    int leg_start_ind, int alpha_v_lower_start_ind, int alpha_v_upper_start_ind,
    int delta_v_lower_start_ind, int delta_v_upper_start_ind,
    double leg_mean, double leg_sd, arma::vec alpha_mean_v, arma::mat alpha_cov_s,
    arma::vec delta_mean_v, arma::mat delta_cov_s,
    int num_iter, int start_iter, int keep_iter, int pos_ind, int neg_ind) {

  vec current_param_val_v = all_param_draws.row(0).t();
  int num_questions = question_num_choices_m1_v.n_elem;
  int num_ind = alpha_v_lower_start_ind - leg_start_ind;
  int max_response_num = (y_star_m.n_cols - 1) / 2;
  uvec num_responses(num_questions + 1);
  num_responses(span(1, num_questions)) = cumsum(question_num_choices_m1_v);
  num_responses(0) = 0;
  mat swap_m(num_iter / 50, num_questions, fill::zeros);

  for (int i = 0; i < num_iter; i++) {
    if (i % 100 == 0) {
      Rcout << i << "\n";
    }

    for (unsigned int j = 0; j < y_star_m.n_rows; j++) {

      int half_alpha_size = question_num_choices_m1_v(question_v(j));

      uvec interested_q_inds =
        linspace<uvec>(question_v(j) * max_response_num,
                       question_v(j) * max_response_num + half_alpha_size - 1,
                       half_alpha_size);
      uvec tmp = {(y_star_m.n_cols - 1) / 2};
      uvec interested_row_ind = {j};
      vec output_v = sample_y_star_m(
        y_star_m.submat(interested_row_ind,
                        join_vert(
                          linspace<uvec>(0, half_alpha_size - 1, half_alpha_size), tmp,
                          linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                         half_alpha_size))).t(),
                                         vote_v(j),
                                         current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
                                         current_param_val_v(
                                           alpha_v_upper_start_ind + interested_q_inds -
                                             half_alpha_size + max_response_num),
                                             current_param_val_v(leg_start_ind + respondent_v(j)),
                                             current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
                                             current_param_val_v(delta_v_upper_start_ind + interested_q_inds -
                                               half_alpha_size + max_response_num));

      y_star_m(j, span(0, half_alpha_size - 1)) = output_v(span(0, half_alpha_size - 1)).t();
      y_star_m(j, span(y_star_m.n_cols - half_alpha_size,
                       y_star_m.n_cols - 1)) =
                         output_v(span(half_alpha_size + 1, 2 * half_alpha_size)).t();
      y_star_m(j, max_response_num) = output_v(half_alpha_size);
    }

    for (unsigned int j = 0; j < num_ind; j++) {
      uvec interested_inds = find(respondent_v == j);
      uvec interested_q_dim_inds(interested_inds.n_elem * max_response_num);
      for (int k = 0; k < interested_inds.n_elem; k++) {
        int question_ind = question_v(interested_inds(k));

        interested_q_dim_inds(
          span(k * max_response_num,
               (k + 1) * max_response_num - 1)) =
                 linspace<uvec>(question_ind * max_response_num,
                                (question_ind + 1) * max_response_num - 1, max_response_num);
      }

      current_param_val_v(leg_start_ind + j) =
        sample_ordinal_utility_probit_beta(
          y_star_m.submat(interested_inds,
                          linspace<uvec>(0, max_response_num - 1, max_response_num)),
                          y_star_m.submat(interested_inds,
                                          linspace<uvec>(max_response_num + 1, y_star_m.n_cols - 1,
                                                         max_response_num)),
                                                         reshape(current_param_val_v(alpha_v_lower_start_ind + interested_q_dim_inds),
                                                                 max_response_num, interested_inds.n_elem).t(),
                                                                 reshape(current_param_val_v(alpha_v_upper_start_ind + interested_q_dim_inds),
                                                                         max_response_num, interested_inds.n_elem).t(),
                                                                         reshape(current_param_val_v(delta_v_lower_start_ind + interested_q_dim_inds),
                                                                                 max_response_num, interested_inds.n_elem).t(),
                                                                                 reshape(current_param_val_v(delta_v_upper_start_ind + interested_q_dim_inds),
                                                                                         max_response_num, interested_inds.n_elem).t(),
                                                                                         leg_mean, leg_sd);
    }

    vec match_var_v(num_questions);
    for (unsigned int j = 0; j < num_questions; j++) {

      int half_alpha_size = question_num_choices_m1_v(j);

      uvec interested_inds = find(question_v == j);
      uvec interested_q_inds =
        linspace<uvec>(j * max_response_num,
                       j * max_response_num + half_alpha_size - 1, half_alpha_size);
      uvec interested_q_inds_upper =
        linspace<uvec>((j + 1) * max_response_num - half_alpha_size,
                       (j + 1) * max_response_num - 1, half_alpha_size);

      vec out_v = sample_ordinal_probit_matched_alpha_intervals_no_flip(
        y_star_m.submat(interested_inds,
                        join_vert(linspace<uvec>(0, half_alpha_size - 1, half_alpha_size),
                                  linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                                 half_alpha_size))),
                                                 current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
                                                 join_vert(current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
                                                           current_param_val_v(delta_v_upper_start_ind + interested_q_inds_upper)),
                                                           alpha_mean_v(span(max_response_num - half_alpha_size,
                                                                             max_response_num + half_alpha_size - 1)),
                                                                             alpha_cov_s(span(max_response_num - half_alpha_size,
                                                                                              max_response_num + half_alpha_size - 1),
                                                                                              span(max_response_num - half_alpha_size,
                                                                                                   max_response_num + half_alpha_size - 1)),
                                                                                                   delta_mean_v, delta_cov_s);
      current_param_val_v(alpha_v_lower_start_ind + interested_q_inds) =
        out_v(span(0, half_alpha_size - 1));
      current_param_val_v(
        alpha_v_upper_start_ind + interested_q_inds_upper) =
          out_v(span(half_alpha_size, 2 * half_alpha_size - 1));
      match_var_v(j) = out_v(out_v.n_elem - 1);
    }

    for (unsigned int j = 0; j < num_questions; j++) {
      uvec interested_inds = find(question_v == j);
      int half_alpha_size = question_num_choices_m1_v(j);

      uvec interested_q_inds =
        linspace<uvec>(j * max_response_num,
                       j * max_response_num + half_alpha_size - 1, half_alpha_size);
      uvec interested_q_inds_upper =
        linspace<uvec>((j + 1) * max_response_num - half_alpha_size,
                       (j + 1) * max_response_num - 1, half_alpha_size);

      vec out_v = sample_ordinal_utility_matched_delta(
        y_star_m.submat(interested_inds,
                        linspace<uvec>(0, half_alpha_size - 1, half_alpha_size)),
                        y_star_m.submat(interested_inds,
                                        linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                                       half_alpha_size)),
                                                       current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
                                                       current_param_val_v(alpha_v_upper_start_ind + interested_q_inds_upper),
                                                       current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
                                                       match_var_v(j),
                                                       delta_mean_v(span(max_response_num - half_alpha_size,
                                                                         max_response_num + half_alpha_size - 1)),
                                                                         delta_cov_s(span(max_response_num - half_alpha_size,
                                                                                          max_response_num + half_alpha_size - 1),
                                                                                          span(max_response_num - half_alpha_size,
                                                                                               max_response_num + half_alpha_size - 1)));
      current_param_val_v(delta_v_lower_start_ind + interested_q_inds) =
        out_v(span(0, half_alpha_size - 1));
      current_param_val_v(
        delta_v_upper_start_ind + interested_q_inds_upper) =
          out_v(span(half_alpha_size, out_v.n_elem - 1));
    }

    if (i > 0 && ((i + 1) % 50 == 0)) {
      for (unsigned int j = 0; j < num_questions; j++) {
        uvec interested_inds = find(question_v == j);
        int half_alpha_size = question_num_choices_m1_v(j);

        uvec interested_q_inds =
          linspace<uvec>(j * max_response_num,
                         j * max_response_num + half_alpha_size - 1,
                         half_alpha_size);

        uvec interested_q_inds_upper =
          linspace<uvec>((j + 1) * max_response_num - half_alpha_size,
                         (j + 1) * max_response_num - 1, half_alpha_size);
        uvec middle_ind = {static_cast<unsigned int>(max_response_num)};

        vec out_v = flip_signs_response(
        // List out_v_info = flip_signs_response_gibbs(
          vote_v(interested_inds),
          // y_star_m.submat(
          //   interested_inds,
          //   join_vert(linspace<uvec>(0, half_alpha_size - 1, half_alpha_size),
          //             middle_ind,
          //             linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
          //                            half_alpha_size))),
                                     current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
                                     current_param_val_v(alpha_v_upper_start_ind + interested_q_inds_upper),
                                     current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
                                     current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
                                     current_param_val_v(delta_v_upper_start_ind + interested_q_inds_upper),
                                     alpha_mean_v(span(max_response_num - half_alpha_size,
                                                       max_response_num + half_alpha_size - 1)),
                                                       alpha_cov_s(span(max_response_num - half_alpha_size,
                                                                        max_response_num + half_alpha_size - 1),
                                                                        span(max_response_num - half_alpha_size,
                                                                             max_response_num + half_alpha_size - 1)),
                                                                             delta_mean_v(span(max_response_num - half_alpha_size,
                                                                                               max_response_num + half_alpha_size - 1)),
                                                                                               delta_cov_s(span(max_response_num - half_alpha_size,
                                                                                                                max_response_num + half_alpha_size - 1),
                                                                                                                span(max_response_num - half_alpha_size,
                                                                                                                     max_response_num + half_alpha_size - 1)));
        // vec out_v = out_v_info[0];
        current_param_val_v(alpha_v_lower_start_ind + interested_q_inds) =
          out_v(span(0, half_alpha_size - 1));
        current_param_val_v(alpha_v_upper_start_ind + interested_q_inds_upper) =
          out_v(span(half_alpha_size, 2 * half_alpha_size - 1));
        current_param_val_v(delta_v_lower_start_ind + interested_q_inds) =
          out_v(span(2 * half_alpha_size, 3 * half_alpha_size - 1));
        current_param_val_v(delta_v_upper_start_ind + interested_q_inds_upper) =
          out_v(span(3 * half_alpha_size, 4 * half_alpha_size - 1));
        if (out_v(out_v.n_elem - 1) > 0.5) {
          vote_v(interested_inds) = half_alpha_size - vote_v(interested_inds);
          y_star_m(interested_inds).zeros();
          for (unsigned int j = 0; j < interested_inds.n_elem; j++) {
            // for (unsigned int k = 0; k < half_alpha_size / 2; k++) {
            // double max_val = y_star_m.row(j).max();
            if (vote_v(interested_inds(j)) == half_alpha_size) {
              y_star_m(interested_inds(j), max_response_num) = 1;
            } else {
              if (randu() < 0.5) {
                y_star_m(interested_inds(j),  vote_v(interested_inds(j))) = 1;
              } else {
                y_star_m(interested_inds(j), y_star_m.n_cols - 1 -  vote_v(interested_inds(j))) = 1;
              }
            }
          }
          swap_m((i + 1) / 50 - 1, j) = 1;

          // y_star_m.submat(interested_inds,
          //                 linspace<uvec>(0, half_alpha_size - 1,
          //                                half_alpha_size));
          // y_star_m.submat(interested_inds,
          //                 linspace<uvec>(y_star_m.n_cols - half_alpha_size,
          //                                y_star_m.n_cols - 1, half_alpha_size));
          //                 join_vert(linspace<uvec>(0, half_alpha_size - 1, half_alpha_size),
          //                           linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
          //                                          half_alpha_size)))
        }
        // vote_v(interested_inds) = out_v(
        //   span(4 * half_alpha_size, out_v.n_elem - 1));
      }
    }


    int post_burn_i = i - start_iter + 1;
    if (i >= start_iter && (fmod(post_burn_i, keep_iter) == 0)) {
      int keep_iter_ind = post_burn_i / keep_iter - 1;
      all_param_draws.row(keep_iter_ind) = current_param_val_v.t();
    }
  }

  return(List::create(Named("param_draws") = all_param_draws,
                      Named("y_star_m") = y_star_m,
                      Named("swap_tracker") = swap_m));
}

// [[Rcpp::export]]
List sample_ordinal_utility_probit_gen_choices_flip_alpha(
    arma::uvec vote_v, arma::uvec respondent_v, arma::uvec question_v,
    arma::uvec question_num_choices_m1_v,
    arma::mat all_param_draws, arma::mat y_star_m,
    int leg_start_ind, int alpha_v_lower_start_ind, int alpha_v_upper_start_ind,
    int delta_v_lower_start_ind, int delta_v_upper_start_ind,
    double leg_mean, double leg_sd, arma::vec alpha_mean_v, arma::mat alpha_cov_s,
    arma::vec delta_mean_v, arma::mat delta_cov_s,
    int num_iter, int start_iter, int keep_iter, int pos_ind, int neg_ind) {

  vec current_param_val_v = all_param_draws.row(0).t();
  int num_questions = question_num_choices_m1_v.n_elem;
  int num_ind = alpha_v_lower_start_ind - leg_start_ind;
  int max_response_num = (y_star_m.n_cols - 1) / 2;
  uvec num_responses(num_questions + 1);
  num_responses(span(1, num_questions)) = cumsum(question_num_choices_m1_v);
  num_responses(0) = 0;

  for (int i = 0; i < num_iter; i++) {
    if (i % 100 == 0) {
      Rcout << i << "\n";
    }

    for (unsigned int j = 0; j < y_star_m.n_rows; j++) {

      int half_alpha_size = question_num_choices_m1_v(question_v(j));

      uvec interested_q_inds =
        linspace<uvec>(question_v(j) * max_response_num,
                       question_v(j) * max_response_num + half_alpha_size - 1,
                       half_alpha_size);
      uvec tmp = {(y_star_m.n_cols - 1) / 2};
      uvec interested_row_ind = {j};
      vec output_v = sample_y_star_m(
        y_star_m.submat(interested_row_ind,
                        join_vert(
                          linspace<uvec>(0, half_alpha_size - 1, half_alpha_size), tmp,
                          linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                         half_alpha_size))).t(),
                                         vote_v(j),
                                         current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
                                         current_param_val_v(
                                           alpha_v_upper_start_ind + interested_q_inds -
                                             half_alpha_size + max_response_num),
                                             current_param_val_v(leg_start_ind + respondent_v(j)),
                                             current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
                                             current_param_val_v(delta_v_upper_start_ind + interested_q_inds -
                                               half_alpha_size + max_response_num));

      y_star_m(j, span(0, half_alpha_size - 1)) = output_v(span(0, half_alpha_size - 1)).t();
      y_star_m(j, span(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1)) =
                         output_v(span(half_alpha_size + 1, 2 * half_alpha_size)).t();
      y_star_m(j, max_response_num) = output_v(half_alpha_size);
    }

    for (unsigned int j = 0; j < num_ind; j++) {
      uvec interested_inds = find(respondent_v == j);
      uvec interested_q_dim_inds(interested_inds.n_elem * max_response_num);
      for (int k = 0; k < interested_inds.n_elem; k++) {
        int question_ind = question_v(interested_inds(k));

        interested_q_dim_inds(
          span(k * max_response_num,
               (k + 1) * max_response_num - 1)) =
                 linspace<uvec>(question_ind * max_response_num,
                                (question_ind + 1) * max_response_num - 1, max_response_num);
      }

      current_param_val_v(leg_start_ind + j) =
        sample_ordinal_utility_probit_beta(
          y_star_m.submat(interested_inds,
                          linspace<uvec>(0, max_response_num - 1, max_response_num)),
                          y_star_m.submat(interested_inds,
                                          linspace<uvec>(max_response_num + 1, y_star_m.n_cols - 1,
                                                         max_response_num)),
                                                         reshape(current_param_val_v(alpha_v_lower_start_ind + interested_q_dim_inds),
                                                                 max_response_num, interested_inds.n_elem).t(),
                                                                 reshape(current_param_val_v(alpha_v_upper_start_ind + interested_q_dim_inds),
                                                                         max_response_num, interested_inds.n_elem).t(),
                                                                         reshape(current_param_val_v(delta_v_lower_start_ind + interested_q_dim_inds),
                                                                                 max_response_num, interested_inds.n_elem).t(),
                                                                                 reshape(current_param_val_v(delta_v_upper_start_ind + interested_q_dim_inds),
                                                                                         max_response_num, interested_inds.n_elem).t(),
                                                                                         leg_mean, leg_sd);
    }

    vec match_var_v(num_questions);
    for (unsigned int j = 0; j < num_questions; j++) {
      int half_alpha_size = question_num_choices_m1_v(j);

      uvec interested_inds = find(question_v == j);
      uvec interested_q_inds =
        linspace<uvec>(j * max_response_num,
                       j * max_response_num + half_alpha_size - 1, half_alpha_size);
      uvec interested_q_inds_upper =
        linspace<uvec>((j + 1) * max_response_num - half_alpha_size,
                       (j + 1) * max_response_num - 1, half_alpha_size);

      vec out_v = sample_ordinal_probit_matched_alpha_intervals(
        y_star_m.submat(interested_inds,
                        join_vert(linspace<uvec>(0, half_alpha_size - 1, half_alpha_size),
                                  linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                                 half_alpha_size))),
                        current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
                        join_vert(current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
                                  current_param_val_v(delta_v_upper_start_ind + interested_q_inds_upper)),
                        alpha_mean_v(span(max_response_num - half_alpha_size,
                                          max_response_num + half_alpha_size - 1)),
                        alpha_cov_s(span(max_response_num - half_alpha_size,
                                         max_response_num + half_alpha_size - 1),
                                    span(max_response_num - half_alpha_size,
                                         max_response_num + half_alpha_size - 1)),
                        delta_mean_v(span(max_response_num - half_alpha_size,
                                         max_response_num + half_alpha_size - 1)),
                        delta_cov_s(span(max_response_num - half_alpha_size,
                                         max_response_num + half_alpha_size - 1),
                                         span(max_response_num - half_alpha_size,
                                              max_response_num + half_alpha_size - 1)));
      current_param_val_v(alpha_v_lower_start_ind + interested_q_inds) =
        out_v(span(0, half_alpha_size - 1));
      current_param_val_v(
        alpha_v_upper_start_ind + interested_q_inds_upper) =
          out_v(span(half_alpha_size, 2 * half_alpha_size - 1));
      match_var_v(j) = out_v(out_v.n_elem - 1);
    }

    for (unsigned int j = 0; j < num_questions; j++) {
      uvec interested_inds = find(question_v == j);
      int half_alpha_size = question_num_choices_m1_v(j);

      uvec interested_q_inds =
        linspace<uvec>(j * max_response_num,
                       j * max_response_num + half_alpha_size - 1, half_alpha_size);
      uvec interested_q_inds_upper =
        linspace<uvec>((j + 1) * max_response_num - half_alpha_size,
                       (j + 1) * max_response_num - 1, half_alpha_size);

      vec out_v = sample_ordinal_utility_matched_delta(
        y_star_m.submat(interested_inds,
                        linspace<uvec>(0, half_alpha_size - 1, half_alpha_size)),
                        y_star_m.submat(interested_inds,
                                        linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                                       half_alpha_size)),
                                                       current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
                                                       current_param_val_v(alpha_v_upper_start_ind + interested_q_inds_upper),
                                                       current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
                                                       match_var_v(j),
                                                       delta_mean_v(span(max_response_num - half_alpha_size,
                                                                         max_response_num + half_alpha_size - 1)),
                                                                         delta_cov_s(span(max_response_num - half_alpha_size,
                                                                                          max_response_num + half_alpha_size - 1),
                                                                                          span(max_response_num - half_alpha_size,
                                                                                               max_response_num + half_alpha_size - 1)));
      current_param_val_v(delta_v_lower_start_ind + interested_q_inds) =
        out_v(span(0, half_alpha_size - 1));
      current_param_val_v(
        delta_v_upper_start_ind + interested_q_inds_upper) =
          out_v(span(half_alpha_size, out_v.n_elem - 1));
    }

    if (pos_ind > -1 && (current_param_val_v(leg_start_ind + pos_ind) < 0)) {
      current_param_val_v = -current_param_val_v;
    }

    if (neg_ind > -1 && pos_ind < 0 && (current_param_val_v(leg_start_ind + neg_ind) > 0)) {
      current_param_val_v = -current_param_val_v;
    }

    int post_burn_i = i - start_iter + 1;
    if (i >= start_iter && (fmod(post_burn_i, keep_iter) == 0)) {
      int keep_iter_ind = post_burn_i / keep_iter - 1;
      all_param_draws.row(keep_iter_ind) = current_param_val_v.t();
    }
  }

  return(List::create(Named("param_draws") = all_param_draws,
                      Named("y_star_m") = y_star_m));
}

// [[Rcpp::export]]
List sample_ordinal_utility_probit_flip(
    arma::uvec vote_v, arma::uvec respondent_v, arma::uvec question_v,
    arma::mat all_param_draws, arma::mat y_star_m,
    int leg_start_ind, int alpha_v_lower_start_ind, int alpha_v_upper_start_ind,
    int delta_v_lower_start_ind, int delta_v_upper_start_ind,
    double leg_mean, double leg_sd, arma::vec alpha_mean_v, arma::mat alpha_cov_s,
    arma::vec delta_mean_v, arma::mat delta_cov_s,
    int num_iter, int start_iter, int keep_iter, int pos_ind, int neg_ind) {

  vec current_param_val_v = all_param_draws.row(0).t();
  int half_alpha_size = (y_star_m.n_cols - 1) / 2;
  int num_questions = (alpha_v_upper_start_ind - alpha_v_lower_start_ind) / half_alpha_size;
  int num_ind = alpha_v_lower_start_ind - leg_start_ind;

  mat match_m(num_iter / 100, num_questions, fill::ones);

  for (int i = 0; i < num_iter; i++) {
    if (i % 100 == 0) {
      Rcout << i << "\n";
    }

    for (int j = 0; j < y_star_m.n_rows; j++) {
      uvec interested_q_inds =
        linspace<uvec>(question_v(j) * half_alpha_size,
                       (question_v(j) + 1) * half_alpha_size - 1, half_alpha_size);
      vec output_v = sample_y_star_m(
        y_star_m.row(j).t(), vote_v(j),
        current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
        current_param_val_v(alpha_v_upper_start_ind + interested_q_inds),
        current_param_val_v(leg_start_ind + respondent_v(j)),
        current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
        current_param_val_v(delta_v_upper_start_ind + interested_q_inds));
      y_star_m.row(j) = output_v.t();
    }

    for (unsigned int j = 0; j < num_ind; j++) {
      uvec interested_inds = find(respondent_v == j);
      uvec interested_q_dim_inds(interested_inds.n_elem * half_alpha_size);
      for (int k = 0; k < interested_inds.n_elem; k++) {
        int question_ind = question_v(interested_inds(k));

        interested_q_dim_inds(
          span(k * half_alpha_size,
               (k + 1) * half_alpha_size - 1)) =
                 linspace<uvec>(question_ind * half_alpha_size,
                                (question_ind + 1) * half_alpha_size - 1, half_alpha_size);
      }

      current_param_val_v(leg_start_ind + j) =
        sample_ordinal_utility_probit_beta(
          y_star_m.submat(interested_inds,
                          linspace<uvec>(0, half_alpha_size - 1, half_alpha_size)),
                          y_star_m.submat(interested_inds,
                                          linspace<uvec>(half_alpha_size + 1, y_star_m.n_cols - 1, half_alpha_size)),
                                          reshape(current_param_val_v(alpha_v_lower_start_ind + interested_q_dim_inds),
                                                  half_alpha_size, interested_inds.n_elem).t(),
                                                  reshape(current_param_val_v(alpha_v_upper_start_ind + interested_q_dim_inds),
                                                          half_alpha_size, interested_inds.n_elem).t(),
                                                          reshape(current_param_val_v(delta_v_lower_start_ind + interested_q_dim_inds),
                                                                  half_alpha_size, interested_inds.n_elem).t(),
                                                                  reshape(current_param_val_v(delta_v_upper_start_ind + interested_q_dim_inds),
                                                                          half_alpha_size, interested_inds.n_elem).t(),
                                                                          leg_mean, leg_sd);
    }

    vec match_var_v(num_questions);
    for (unsigned int j = 0; j < num_questions; j++) {

      uvec interested_inds = find(question_v == j);
      uvec interested_q_inds =
        linspace<uvec>(j * half_alpha_size,
                       (j + 1) * half_alpha_size - 1, half_alpha_size);
      vec out_v = sample_ordinal_probit_matched_alpha_intervals(
        y_star_m.submat(interested_inds,
                        join_vert(linspace<uvec>(0, half_alpha_size - 1, half_alpha_size),
                                  linspace<uvec>(half_alpha_size + 1, y_star_m.n_cols - 1, half_alpha_size))),
                                  current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
                                  join_vert(current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
                                            current_param_val_v(delta_v_upper_start_ind + interested_q_inds)),
                                            alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s);
      current_param_val_v(alpha_v_lower_start_ind + interested_q_inds) =
        out_v(span(0, half_alpha_size - 1));
      current_param_val_v(alpha_v_upper_start_ind + interested_q_inds) =
        out_v(span(half_alpha_size, 2 * half_alpha_size - 1));
      match_var_v(j) = out_v(out_v.n_elem - 1);
    }

    for (unsigned int j = 0; j < num_questions; j++) {
      uvec interested_inds = find(question_v == j);
      uvec interested_q_inds =
        linspace<uvec>(j * half_alpha_size,
                       (j + 1) * half_alpha_size - 1, half_alpha_size);

      vec out_v = sample_ordinal_utility_matched_delta(
        y_star_m.submat(interested_inds,
                        linspace<uvec>(0, half_alpha_size - 1, half_alpha_size)),
                        y_star_m.submat(interested_inds,
                                        linspace<uvec>(half_alpha_size + 1, y_star_m.n_cols - 1, half_alpha_size)),
                                        current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
                                        current_param_val_v(alpha_v_upper_start_ind + interested_q_inds),
                                        current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
                                        match_var_v(j), delta_mean_v, delta_cov_s);
      current_param_val_v(delta_v_lower_start_ind + interested_q_inds) =
        out_v(span(0, half_alpha_size - 1));
      current_param_val_v(delta_v_upper_start_ind + interested_q_inds) =
        out_v(span(half_alpha_size, out_v.n_elem - 1));
    }

    if (i > 0 && ((i + 1) % 100 == 0)) {
      for (unsigned int j = 0; j < num_questions; j++) {
        uvec interested_inds = find(question_v == j);
        uvec interested_q_inds =
          linspace<uvec>(j * half_alpha_size,
                         (j + 1) * half_alpha_size - 1, half_alpha_size);

        vec out_v = flip_signs(
          vote_v(interested_inds),
          current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
          current_param_val_v(alpha_v_upper_start_ind + interested_q_inds),
          current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
          current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
          current_param_val_v(delta_v_upper_start_ind + interested_q_inds));
        match_m(i / 100, j) = sign(out_v(0) *
          current_param_val_v(alpha_v_lower_start_ind + interested_q_inds(0)));
        current_param_val_v(alpha_v_lower_start_ind + interested_q_inds) =
          out_v(span(0, half_alpha_size - 1));
        current_param_val_v(alpha_v_upper_start_ind + interested_q_inds) =
          out_v(span(half_alpha_size, 2 * half_alpha_size - 1));
        current_param_val_v(delta_v_lower_start_ind + interested_q_inds) =
          out_v(span(2 * half_alpha_size, 3 * half_alpha_size - 1));
        current_param_val_v(delta_v_upper_start_ind + interested_q_inds) =
          out_v(span(3 * half_alpha_size, 4 * half_alpha_size - 1));
      }
    }

    if (pos_ind > -1 && (current_param_val_v(leg_start_ind + pos_ind) < 0)) {
      current_param_val_v = -current_param_val_v;
    }

    if (neg_ind > -1 && pos_ind < 0 && (current_param_val_v(leg_start_ind + neg_ind) > 0)) {
      current_param_val_v = -current_param_val_v;
    }

    int post_burn_i = i - start_iter + 1;
    if (i >= start_iter && (fmod(post_burn_i, keep_iter) == 0)) {
      int keep_iter_ind = post_burn_i / keep_iter - 1;
      all_param_draws.row(keep_iter_ind) = current_param_val_v.t();
    }
  }

  return(List::create(Named("param_draws") = all_param_draws,
                      Named("y_star_m") = y_star_m,
                      Named("flip_tracker") = match_m));
}

// [[Rcpp::export]]
List sample_ordinal_utility_probit_flip_prior(
    arma::uvec vote_v, arma::uvec respondent_v, arma::uvec question_v,
    arma::mat all_param_draws, arma::mat y_star_m,
    int leg_start_ind, int alpha_v_lower_start_ind, int alpha_v_upper_start_ind,
    int delta_v_lower_start_ind, int delta_v_upper_start_ind,
    double leg_mean, double leg_sd, arma::vec alpha_mean_v, arma::mat alpha_cov_s,
    arma::vec delta_mean_v, arma::mat delta_cov_s,
    int num_iter, int start_iter, int keep_iter, int pos_ind, int neg_ind) {

  vec current_param_val_v = all_param_draws.row(0).t();
  int half_alpha_size = (y_star_m.n_cols - 1) / 2;
  int num_questions = (alpha_v_upper_start_ind - alpha_v_lower_start_ind) / half_alpha_size;
  int num_ind = alpha_v_lower_start_ind - leg_start_ind;

  for (int i = 0; i < num_iter; i++) {
    if (i % 100 == 0) {
      Rcout << i << "\n";
    }

    for (int j = 0; j < y_star_m.n_rows; j++) {
      uvec interested_q_inds =
        linspace<uvec>(question_v(j) * half_alpha_size,
                       (question_v(j) + 1) * half_alpha_size - 1, half_alpha_size);
      vec output_v = sample_y_star_m(
        y_star_m.row(j).t(), vote_v(j),
        current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
        current_param_val_v(alpha_v_upper_start_ind + interested_q_inds),
        current_param_val_v(leg_start_ind + respondent_v(j)),
        current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
        current_param_val_v(delta_v_upper_start_ind + interested_q_inds));
      y_star_m.row(j) = output_v.t();
    }


    for (unsigned int j = 0; j < num_ind; j++) {
      uvec interested_inds = find(respondent_v == j);
      uvec interested_q_dim_inds(interested_inds.n_elem * half_alpha_size);
      for (int k = 0; k < interested_inds.n_elem; k++) {
        int question_ind = question_v(interested_inds(k));

        interested_q_dim_inds(
          span(k * half_alpha_size,
               (k + 1) * half_alpha_size - 1)) =
                 linspace<uvec>(question_ind * half_alpha_size,
                                (question_ind + 1) * half_alpha_size - 1, half_alpha_size);
      }

      current_param_val_v(leg_start_ind + j) =
        sample_ordinal_utility_probit_beta(
          y_star_m.submat(interested_inds,
                          linspace<uvec>(0, half_alpha_size - 1, half_alpha_size)),
                          y_star_m.submat(interested_inds,
                                          linspace<uvec>(half_alpha_size + 1, y_star_m.n_cols - 1, half_alpha_size)),
                                          reshape(current_param_val_v(alpha_v_lower_start_ind + interested_q_dim_inds),
                                                  half_alpha_size, interested_inds.n_elem).t(),
                                                  reshape(current_param_val_v(alpha_v_upper_start_ind + interested_q_dim_inds),
                                                          half_alpha_size, interested_inds.n_elem).t(),
                                                          reshape(current_param_val_v(delta_v_lower_start_ind + interested_q_dim_inds),
                                                                  half_alpha_size, interested_inds.n_elem).t(),
                                                                  reshape(current_param_val_v(delta_v_upper_start_ind + interested_q_dim_inds),
                                                                          half_alpha_size, interested_inds.n_elem).t(),
                                                                          leg_mean, leg_sd);
    }

    vec match_var_v(num_questions);
    for (unsigned int j = 0; j < num_questions; j++) {

      uvec interested_inds = find(question_v == j);
      uvec interested_q_inds =
        linspace<uvec>(j * half_alpha_size,
                       (j + 1) * half_alpha_size - 1, half_alpha_size);
      vec out_v = sample_ordinal_probit_matched_alpha_intervals(
        y_star_m.submat(interested_inds,
                        join_vert(linspace<uvec>(0, half_alpha_size - 1, half_alpha_size),
                                  linspace<uvec>(half_alpha_size + 1, y_star_m.n_cols - 1, half_alpha_size))),
                                  current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
                                  join_vert(current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
                                            current_param_val_v(delta_v_upper_start_ind + interested_q_inds)),
                                            alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s);
      current_param_val_v(alpha_v_lower_start_ind + interested_q_inds) =
        out_v(span(0, half_alpha_size - 1));
      current_param_val_v(alpha_v_upper_start_ind + interested_q_inds) =
        out_v(span(half_alpha_size, 2 * half_alpha_size - 1));
      match_var_v(j) = out_v(out_v.n_elem - 1);
    }

    for (unsigned int j = 0; j < num_questions; j++) {
      uvec interested_inds = find(question_v == j);
      uvec interested_q_inds =
        linspace<uvec>(j * half_alpha_size,
                       (j + 1) * half_alpha_size - 1, half_alpha_size);

      vec out_v = sample_ordinal_utility_matched_delta(
        y_star_m.submat(interested_inds,
                        linspace<uvec>(0, half_alpha_size - 1, half_alpha_size)),
                        y_star_m.submat(interested_inds,
                                        linspace<uvec>(half_alpha_size + 1, y_star_m.n_cols - 1, half_alpha_size)),
                                        current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
                                        current_param_val_v(alpha_v_upper_start_ind + interested_q_inds),
                                        current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
                                        match_var_v(j), delta_mean_v, delta_cov_s);
      current_param_val_v(delta_v_lower_start_ind + interested_q_inds) =
        out_v(span(0, half_alpha_size - 1));
      current_param_val_v(delta_v_upper_start_ind + interested_q_inds) =
        out_v(span(half_alpha_size, out_v.n_elem - 1));
    }

    if (i > 0 && ((i + 1) % 100 == 0)) {
      for (unsigned int j = 0; j < num_questions; j++) {
        uvec interested_inds = find(question_v == j);
        uvec interested_q_inds =
          linspace<uvec>(j * half_alpha_size,
                         (j + 1) * half_alpha_size - 1, half_alpha_size);

        vec out_v = flip_signs_prior(
          vote_v(interested_inds),
          current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
          current_param_val_v(alpha_v_upper_start_ind + interested_q_inds),
          current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
          current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
          current_param_val_v(delta_v_upper_start_ind + interested_q_inds),
          match_var_v(j), alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s);
        current_param_val_v(alpha_v_lower_start_ind + interested_q_inds) =
          out_v(span(0, half_alpha_size - 1));
        current_param_val_v(alpha_v_upper_start_ind + interested_q_inds) =
          out_v(span(half_alpha_size, 2 * half_alpha_size - 1));
        current_param_val_v(delta_v_lower_start_ind + interested_q_inds) =
          out_v(span(2 * half_alpha_size, 3 * half_alpha_size - 1));
        current_param_val_v(delta_v_upper_start_ind + interested_q_inds) =
          out_v(span(3 * half_alpha_size, 4 * half_alpha_size - 1));
      }
    }

    int post_burn_i = i - start_iter + 1;
    if (i >= start_iter && (fmod(post_burn_i, keep_iter) == 0)) {
      int keep_iter_ind = post_burn_i / keep_iter - 1;
      all_param_draws.row(keep_iter_ind) = current_param_val_v.t();
    }
  }

  return(List::create(Named("param_draws") = all_param_draws,
                      Named("y_star_m") = y_star_m));
}

// [[Rcpp::export]]
List sample_ordinal_utility_probit_gen_choices_flip_beta_parallel(
      arma::uvec vote_v, arma::uvec respondent_v, arma::uvec question_v,
      arma::uvec question_num_choices_m1_v,
      arma::mat all_param_draws, arma::mat y_star_m,
      int leg_start_ind, int alpha_v_lower_start_ind, int alpha_v_upper_start_ind,
      int delta_v_lower_start_ind, int delta_v_upper_start_ind,
      double leg_mean, double leg_sd, arma::vec alpha_mean_v, arma::mat alpha_cov_s,
      arma::vec delta_mean_v, arma::mat delta_cov_s,
      int num_iter, int start_iter, int keep_iter, int pos_ind, int neg_ind,
      arma::uvec flip_beta_v, double flip_beta_sd, int num_cores) {

    vec current_param_val_v = all_param_draws.row(0).t();
    int num_questions = question_num_choices_m1_v.n_elem;
    int num_ind = alpha_v_lower_start_ind - leg_start_ind;
    int max_response_num = (y_star_m.n_cols - 1) / 2;
    uvec num_responses(num_questions + 1);
    num_responses(span(1, num_questions)) = cumsum(question_num_choices_m1_v);
    num_responses(0) = 0;
    mat swap_m(num_iter / 50, num_questions, fill::zeros);
    vec flip_beta_count(num_ind, fill::zeros);
    mat log_ll_m(num_iter / 50, vote_v.n_elem, fill::zeros);

    vec mean_prob(vote_v.n_elem);
    mean_prob.fill(-datum::inf);
    vec mean_log_prob(vote_v.n_elem, fill::zeros);
    vec log_prob_var(vote_v.n_elem, fill::zeros);
    vec y_star_log_ll(all_param_draws.n_rows);

    for (int i = 0; i < num_iter; i++) {
      if (i % 100 == 0) {
        Rcout << i << "\n";
      }

      for (unsigned int j = 0; j < y_star_m.n_rows; j++) {

        int half_alpha_size = question_num_choices_m1_v(question_v(j));

        uvec interested_q_inds =
          linspace<uvec>(question_v(j) * max_response_num,
                         question_v(j) * max_response_num + half_alpha_size - 1,
                         half_alpha_size);
        uvec tmp = {(y_star_m.n_cols - 1) / 2};
        uvec interested_row_ind = {j};
        vec output_v = sample_y_star_m(
          y_star_m.submat(interested_row_ind,
                          join_vert(
                            linspace<uvec>(0, half_alpha_size - 1, half_alpha_size), tmp,
                            linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                           half_alpha_size))).t(),
                                           vote_v(j),
                                           current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
                                           current_param_val_v(
                                             alpha_v_upper_start_ind + interested_q_inds -
                                               half_alpha_size + max_response_num),
                                               current_param_val_v(leg_start_ind + respondent_v(j)),
                                               current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
                                               current_param_val_v(delta_v_upper_start_ind + interested_q_inds -
                                                 half_alpha_size + max_response_num));

        y_star_m(j, span(0, half_alpha_size - 1)) = output_v(span(0, half_alpha_size - 1)).t();
        y_star_m(j, span(y_star_m.n_cols - half_alpha_size,
                         y_star_m.n_cols - 1)) =
                           output_v(span(half_alpha_size + 1, 2 * half_alpha_size)).t();
        y_star_m(j, max_response_num) = output_v(half_alpha_size);
      }

      for (unsigned int j = 0; j < num_ind; j++) {
        uvec interested_inds = find(respondent_v == j);
        uvec interested_q_dim_inds(interested_inds.n_elem * max_response_num);
        for (int k = 0; k < interested_inds.n_elem; k++) {
          int question_ind = question_v(interested_inds(k));

          interested_q_dim_inds(
            span(k * max_response_num,
                 (k + 1) * max_response_num - 1)) =
                   linspace<uvec>(question_ind * max_response_num,
                                  (question_ind + 1) * max_response_num - 1, max_response_num);
        }

        current_param_val_v(leg_start_ind + j) =
          sample_ordinal_utility_probit_beta(
            y_star_m.submat(interested_inds,
                            linspace<uvec>(0, max_response_num - 1, max_response_num)),
                            y_star_m.submat(interested_inds,
                                            linspace<uvec>(max_response_num + 1, y_star_m.n_cols - 1,
                                                           max_response_num)),
                                                           reshape(current_param_val_v(alpha_v_lower_start_ind + interested_q_dim_inds),
                                                                   max_response_num, interested_inds.n_elem).t(),
                                                                   reshape(current_param_val_v(alpha_v_upper_start_ind + interested_q_dim_inds),
                                                                           max_response_num, interested_inds.n_elem).t(),
                                                                           reshape(current_param_val_v(delta_v_lower_start_ind + interested_q_dim_inds),
                                                                                   max_response_num, interested_inds.n_elem).t(),
                                                                                   reshape(current_param_val_v(delta_v_upper_start_ind + interested_q_dim_inds),
                                                                                           max_response_num, interested_inds.n_elem).t(),
                                                                                           leg_mean, leg_sd);
      }

      vec match_var_v(num_questions);
      for (unsigned int j = 0; j < num_questions; j++) {

        int half_alpha_size = question_num_choices_m1_v(j);

        uvec interested_inds = find(question_v == j);
        uvec interested_q_inds =
          linspace<uvec>(j * max_response_num,
                         j * max_response_num + half_alpha_size - 1, half_alpha_size);
        uvec interested_q_inds_upper =
          linspace<uvec>((j + 1) * max_response_num - half_alpha_size,
                         (j + 1) * max_response_num - 1, half_alpha_size);

        vec out_v = sample_ordinal_probit_matched_alpha_intervals_no_flip(
          y_star_m.submat(interested_inds,
                          join_vert(linspace<uvec>(0, half_alpha_size - 1, half_alpha_size),
                                    linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                                   half_alpha_size))),
                                                   current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
                                                   join_vert(current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
                                                             current_param_val_v(delta_v_upper_start_ind + interested_q_inds_upper)),
                                                             alpha_mean_v(span(max_response_num - half_alpha_size,
                                                                               max_response_num + half_alpha_size - 1)),
                                                                               alpha_cov_s(span(max_response_num - half_alpha_size,
                                                                                                max_response_num + half_alpha_size - 1),
                                                                                                span(max_response_num - half_alpha_size,
                                                                                                     max_response_num + half_alpha_size - 1)),
                                                                                                     delta_mean_v, delta_cov_s);
        current_param_val_v(alpha_v_lower_start_ind + interested_q_inds) =
          out_v(span(0, half_alpha_size - 1));
        current_param_val_v(
          alpha_v_upper_start_ind + interested_q_inds_upper) =
            out_v(span(half_alpha_size, 2 * half_alpha_size - 1));
        match_var_v(j) = out_v(out_v.n_elem - 1);
      }

      for (unsigned int j = 0; j < num_questions; j++) {
        uvec interested_inds = find(question_v == j);
        int half_alpha_size = question_num_choices_m1_v(j);

        uvec interested_q_inds =
          linspace<uvec>(j * max_response_num,
                         j * max_response_num + half_alpha_size - 1, half_alpha_size);
        uvec interested_q_inds_upper =
          linspace<uvec>((j + 1) * max_response_num - half_alpha_size,
                         (j + 1) * max_response_num - 1, half_alpha_size);

        vec out_v = sample_ordinal_utility_matched_delta(
          y_star_m.submat(interested_inds,
                          linspace<uvec>(0, half_alpha_size - 1, half_alpha_size)),
                          y_star_m.submat(interested_inds,
                                          linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                                         half_alpha_size)),
                                                         current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
                                                         current_param_val_v(alpha_v_upper_start_ind + interested_q_inds_upper),
                                                         current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
                                                         match_var_v(j),
                                                         delta_mean_v(span(max_response_num - half_alpha_size,
                                                                           max_response_num + half_alpha_size - 1)),
                                                                           delta_cov_s(span(max_response_num - half_alpha_size,
                                                                                            max_response_num + half_alpha_size - 1),
                                                                                            span(max_response_num - half_alpha_size,
                                                                                                 max_response_num + half_alpha_size - 1)));
        current_param_val_v(delta_v_lower_start_ind + interested_q_inds) =
          out_v(span(0, half_alpha_size - 1));
        current_param_val_v(
          delta_v_upper_start_ind + interested_q_inds_upper) =
            out_v(span(half_alpha_size, out_v.n_elem - 1));
      }

      if (i > 0 && ((i + 1) % 50 == 0)) {

        uvec interested_inds;
        if (flip_beta_v.n_elem < num_ind) {
          interested_inds = find(respondent_v == flip_beta_v(0));
          for (unsigned int i = 1; i < flip_beta_v.n_elem; i++) {
            interested_inds = join_vert(
              interested_inds, find(respondent_v == flip_beta_v(i)));
          }
        } else {
          interested_inds = linspace<uvec>(
            0, vote_v.n_elem - 1, vote_v.n_elem);
        }

        // List out_info = flip_signs_beta_parallel(
        //   vote_v(interested_inds),
        //   reshape(current_param_val_v(span(
        //       alpha_v_lower_start_ind, alpha_v_lower_start_ind +
        //         max_response_num * num_questions - 1)),
        //       max_response_num, num_questions),
        //   reshape(current_param_val_v(span(
        //       alpha_v_upper_start_ind, alpha_v_upper_start_ind +
        //         max_response_num * num_questions - 1)),
        //       max_response_num, num_questions),
        //       current_param_val_v(span(leg_start_ind, leg_start_ind + num_ind - 1)),
        //   reshape(current_param_val_v(span(
        //       delta_v_lower_start_ind, delta_v_lower_start_ind +
        //         max_response_num * num_questions - 1)),
        //         max_response_num, num_questions),
        //   reshape(current_param_val_v(span(
        //       delta_v_upper_start_ind, delta_v_upper_start_ind +
        //               max_response_num * num_questions - 1)),
        //               max_response_num, num_questions),
        //   leg_mean, leg_sd, flip_beta_sd,
        //   respondent_v(interested_inds), flip_beta_v,
        //   question_v(interested_inds), question_num_choices_m1_v,
        //   num_ind, pos_ind, neg_ind, num_cores);

        List out_info = flip_signs_beta_parallel_2(
          vote_v(interested_inds),
          current_param_val_v(span(
              alpha_v_lower_start_ind, alpha_v_lower_start_ind +
                max_response_num * num_questions - 1)),
                current_param_val_v(span(
                    alpha_v_upper_start_ind, alpha_v_upper_start_ind +
                      max_response_num * num_questions - 1)),
                      current_param_val_v(span(leg_start_ind, leg_start_ind + num_ind - 1)),
                      current_param_val_v(span(
                          delta_v_lower_start_ind, delta_v_lower_start_ind +
                            max_response_num * num_questions - 1)),
                            current_param_val_v(span(
                                delta_v_upper_start_ind, delta_v_upper_start_ind +
                                  max_response_num * num_questions - 1)),
                                  leg_mean, leg_sd, flip_beta_sd,
                                  respondent_v(interested_inds), flip_beta_v,
                                  question_v(interested_inds), question_num_choices_m1_v,
                                  max_response_num, pos_ind, neg_ind, num_cores);

        vec out_v = out_info[0];
        uvec ind_list =
          linspace<uvec>(leg_start_ind, leg_start_ind + num_ind - 1, num_ind);
        current_param_val_v(ind_list) = out_v;

        uvec interested_row = {static_cast<unsigned int>(i) / 50};
        vec out_v_1 = out_info[1];
        log_ll_m(interested_row, interested_inds) = out_v_1.t();

        vec out_v_2 = out_info[2];
        flip_beta_count += out_v_2;
        // vec log_ll = out_info[1];

        }
        // vote_v(interested_inds) = out_v(
        //   span(4 * half_alpha_size, out_v.n_elem - 1));

      if (pos_ind > -1 && (current_param_val_v(leg_start_ind + pos_ind) < 0)) {
        current_param_val_v(span(0, alpha_v_lower_start_ind - 1)) =
          -current_param_val_v(span(0, alpha_v_lower_start_ind - 1));
          current_param_val_v(span(delta_v_lower_start_ind, current_param_val_v.n_elem - 1)) =
          -current_param_val_v(span(delta_v_lower_start_ind, current_param_val_v.n_elem - 1));
          for (unsigned int j = 0; j < num_questions; j++) {
            for (unsigned int k = 0; k < max_response_num; k++) {
              double tmp = current_param_val_v(alpha_v_lower_start_ind + j * max_response_num + k);
              current_param_val_v(alpha_v_lower_start_ind + j * max_response_num + k) =
                -current_param_val_v(alpha_v_upper_start_ind + (j + 1) * max_response_num - 1 - k);
                current_param_val_v(alpha_v_upper_start_ind + (j + 1) * max_response_num - 1 - k) =
                -tmp;
            }
          }
      }

      if (neg_ind > -1 && pos_ind < 0 && (current_param_val_v(leg_start_ind + neg_ind) > 0)) {
        current_param_val_v(span(0, alpha_v_lower_start_ind - 1)) =
          -current_param_val_v(span(0, alpha_v_lower_start_ind - 1));
          current_param_val_v(span(delta_v_lower_start_ind, current_param_val_v.n_elem - 1)) =
          -current_param_val_v(span(delta_v_lower_start_ind, current_param_val_v.n_elem - 1));
          for (unsigned int j = 0; j < num_questions; j++) {
            for (unsigned int k = 0; k < max_response_num; k++) {
              double tmp = current_param_val_v(alpha_v_lower_start_ind + j * max_response_num + k);
              current_param_val_v(alpha_v_lower_start_ind + j * max_response_num + k) =
                -current_param_val_v(alpha_v_upper_start_ind + (j + 1) * max_response_num - 1 - k);
                current_param_val_v(alpha_v_upper_start_ind + (j + 1) * max_response_num - 1 - k) =
                -tmp;
            }
          }
      }

      int post_burn_i = i - start_iter + 1;
      if (i >= start_iter && (fmod(post_burn_i, keep_iter) == 0)) {
        int keep_iter_ind = post_burn_i / keep_iter - 1;
        all_param_draws.row(keep_iter_ind) = current_param_val_v.t();
        y_star_log_ll(keep_iter_ind) =
          calc_log_ll_y_star(
            vote_v,
            y_star_m, reshape(current_param_val_v(span(
                alpha_v_lower_start_ind, alpha_v_lower_start_ind +
                  max_response_num * num_questions - 1)),
                  max_response_num, num_questions),
                  reshape(current_param_val_v(span(
                      alpha_v_upper_start_ind, alpha_v_upper_start_ind +
                        max_response_num * num_questions - 1)),
                        max_response_num, num_questions),
                        current_param_val_v(span(leg_start_ind, leg_start_ind + num_ind - 1)),
                        reshape(current_param_val_v(span(
                            delta_v_lower_start_ind, delta_v_lower_start_ind +
                              max_response_num * num_questions - 1)),
                              max_response_num, num_questions),
                              reshape(current_param_val_v(span(
                                  delta_v_upper_start_ind, delta_v_upper_start_ind +
                                    max_response_num * num_questions - 1)),
                                    max_response_num, num_questions),
            respondent_v, question_v, question_num_choices_m1_v);

      }
    }

    return(List::create(Named("param_draws") = all_param_draws,
                        Named("y_star_m") = y_star_m,
                        Named("swap_tracker") = flip_beta_count,
                        Named("y_star_m_log_ll") = y_star_log_ll,
                        Named("log_ll_m") = log_ll_m));
}

// [[Rcpp::export]]
List sample_ordinal_utility_probit_gen_choices_flip_beta(
      arma::uvec vote_v, arma::uvec respondent_v, arma::uvec question_v,
      arma::uvec question_num_choices_m1_v,
      arma::mat all_param_draws, arma::mat y_star_m,
      int leg_start_ind, int alpha_v_lower_start_ind, int alpha_v_upper_start_ind,
      int delta_v_lower_start_ind, int delta_v_upper_start_ind,
      double leg_mean, double leg_sd, arma::vec alpha_mean_v, arma::mat alpha_cov_s,
      arma::vec delta_mean_v, arma::mat delta_cov_s,
      int num_iter, int start_iter, int keep_iter, int pos_ind, int neg_ind,
      arma::vec flip_beta_v, double flip_beta_sd) {

    vec current_param_val_v = all_param_draws.row(0).t();
    int num_questions = question_num_choices_m1_v.n_elem;
    int num_ind = alpha_v_lower_start_ind - leg_start_ind;
    int max_response_num = (y_star_m.n_cols - 1) / 2;
    uvec num_responses(num_questions + 1);
    num_responses(span(1, num_questions)) = cumsum(question_num_choices_m1_v);
    num_responses(0) = 0;
    mat swap_m(num_iter / 50, num_questions, fill::zeros);
    ivec flip_beta_count(num_ind, fill::zeros);

    for (int i = 0; i < num_iter; i++) {
      if (i % 100 == 0) {
        Rcout << i << "\n";
      }

      for (unsigned int j = 0; j < y_star_m.n_rows; j++) {

        int half_alpha_size = question_num_choices_m1_v(question_v(j));

        uvec interested_q_inds =
          linspace<uvec>(question_v(j) * max_response_num,
                         question_v(j) * max_response_num + half_alpha_size - 1,
                         half_alpha_size);
        uvec tmp = {(y_star_m.n_cols - 1) / 2};
        uvec interested_row_ind = {j};
        vec output_v = sample_y_star_m(
          y_star_m.submat(interested_row_ind,
                          join_vert(
                            linspace<uvec>(0, half_alpha_size - 1, half_alpha_size), tmp,
                            linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                           half_alpha_size))).t(),
                                           vote_v(j),
                                           current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
                                           current_param_val_v(
                                             alpha_v_upper_start_ind + interested_q_inds -
                                               half_alpha_size + max_response_num),
                                               current_param_val_v(leg_start_ind + respondent_v(j)),
                                               current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
                                               current_param_val_v(delta_v_upper_start_ind + interested_q_inds -
                                                 half_alpha_size + max_response_num));

        y_star_m(j, span(0, half_alpha_size - 1)) = output_v(span(0, half_alpha_size - 1)).t();
        y_star_m(j, span(y_star_m.n_cols - half_alpha_size,
                         y_star_m.n_cols - 1)) =
                           output_v(span(half_alpha_size + 1, 2 * half_alpha_size)).t();
        y_star_m(j, max_response_num) = output_v(half_alpha_size);
      }

      for (unsigned int j = 0; j < num_ind; j++) {
        uvec interested_inds = find(respondent_v == j);
        uvec interested_q_dim_inds(interested_inds.n_elem * max_response_num);
        for (int k = 0; k < interested_inds.n_elem; k++) {
          int question_ind = question_v(interested_inds(k));

          interested_q_dim_inds(
            span(k * max_response_num,
                 (k + 1) * max_response_num - 1)) =
                   linspace<uvec>(question_ind * max_response_num,
                                  (question_ind + 1) * max_response_num - 1, max_response_num);
        }

        current_param_val_v(leg_start_ind + j) =
          sample_ordinal_utility_probit_beta(
            y_star_m.submat(interested_inds,
                            linspace<uvec>(0, max_response_num - 1, max_response_num)),
                            y_star_m.submat(interested_inds,
                                            linspace<uvec>(max_response_num + 1, y_star_m.n_cols - 1,
                                                           max_response_num)),
                                                           reshape(current_param_val_v(alpha_v_lower_start_ind + interested_q_dim_inds),
                                                                   max_response_num, interested_inds.n_elem).t(),
                                                                   reshape(current_param_val_v(alpha_v_upper_start_ind + interested_q_dim_inds),
                                                                           max_response_num, interested_inds.n_elem).t(),
                                                                           reshape(current_param_val_v(delta_v_lower_start_ind + interested_q_dim_inds),
                                                                                   max_response_num, interested_inds.n_elem).t(),
                                                                                   reshape(current_param_val_v(delta_v_upper_start_ind + interested_q_dim_inds),
                                                                                           max_response_num, interested_inds.n_elem).t(),
                                                                                           leg_mean, leg_sd);
      }

      vec match_var_v(num_questions);
      for (unsigned int j = 0; j < num_questions; j++) {

        int half_alpha_size = question_num_choices_m1_v(j);

        uvec interested_inds = find(question_v == j);
        uvec interested_q_inds =
          linspace<uvec>(j * max_response_num,
                         j * max_response_num + half_alpha_size - 1, half_alpha_size);
        uvec interested_q_inds_upper =
          linspace<uvec>((j + 1) * max_response_num - half_alpha_size,
                         (j + 1) * max_response_num - 1, half_alpha_size);

        vec out_v = sample_ordinal_probit_matched_alpha_intervals_no_flip(
          y_star_m.submat(interested_inds,
                          join_vert(linspace<uvec>(0, half_alpha_size - 1, half_alpha_size),
                                    linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                                   half_alpha_size))),
                                                   current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
                                                   join_vert(current_param_val_v(delta_v_lower_start_ind + interested_q_inds),
                                                             current_param_val_v(delta_v_upper_start_ind + interested_q_inds_upper)),
                                                             alpha_mean_v(span(max_response_num - half_alpha_size,
                                                                               max_response_num + half_alpha_size - 1)),
                                                                               alpha_cov_s(span(max_response_num - half_alpha_size,
                                                                                                max_response_num + half_alpha_size - 1),
                                                                                                span(max_response_num - half_alpha_size,
                                                                                                     max_response_num + half_alpha_size - 1)),
                                                                                                     delta_mean_v, delta_cov_s);
        current_param_val_v(alpha_v_lower_start_ind + interested_q_inds) =
          out_v(span(0, half_alpha_size - 1));
        current_param_val_v(
          alpha_v_upper_start_ind + interested_q_inds_upper) =
            out_v(span(half_alpha_size, 2 * half_alpha_size - 1));
        match_var_v(j) = out_v(out_v.n_elem - 1);
      }

      for (unsigned int j = 0; j < num_questions; j++) {
        uvec interested_inds = find(question_v == j);
        int half_alpha_size = question_num_choices_m1_v(j);

        uvec interested_q_inds =
          linspace<uvec>(j * max_response_num,
                         j * max_response_num + half_alpha_size - 1, half_alpha_size);
        uvec interested_q_inds_upper =
          linspace<uvec>((j + 1) * max_response_num - half_alpha_size,
                         (j + 1) * max_response_num - 1, half_alpha_size);

        vec out_v = sample_ordinal_utility_matched_delta(
          y_star_m.submat(interested_inds,
                          linspace<uvec>(0, half_alpha_size - 1, half_alpha_size)),
                          y_star_m.submat(interested_inds,
                                          linspace<uvec>(y_star_m.n_cols - half_alpha_size, y_star_m.n_cols - 1,
                                                         half_alpha_size)),
                                                         current_param_val_v(alpha_v_lower_start_ind + interested_q_inds),
                                                         current_param_val_v(alpha_v_upper_start_ind + interested_q_inds_upper),
                                                         current_param_val_v(leg_start_ind + respondent_v(interested_inds)),
                                                         match_var_v(j),
                                                         delta_mean_v(span(max_response_num - half_alpha_size,
                                                                           max_response_num + half_alpha_size - 1)),
                                                                           delta_cov_s(span(max_response_num - half_alpha_size,
                                                                                            max_response_num + half_alpha_size - 1),
                                                                                            span(max_response_num - half_alpha_size,
                                                                                                 max_response_num + half_alpha_size - 1)));
        current_param_val_v(delta_v_lower_start_ind + interested_q_inds) =
          out_v(span(0, half_alpha_size - 1));
        current_param_val_v(
          delta_v_upper_start_ind + interested_q_inds_upper) =
            out_v(span(half_alpha_size, out_v.n_elem - 1));
      }

      if (i > 0 && ((i + 1) % 50 == 0)) {
        // for (unsigned int j = 0; j < num_ind; j++) {
        for (unsigned int j : flip_beta_v) {
          uvec interested_inds = find(respondent_v == j);
          uvec interested_q_dim_inds(interested_inds.n_elem * max_response_num);
          for (int k = 0; k < interested_inds.n_elem; k++) {
            int question_ind = question_v(interested_inds(k));

            interested_q_dim_inds(
              span(k * max_response_num,
                   (k + 1) * max_response_num - 1)) =
                     linspace<uvec>(question_ind * max_response_num,
                                    (question_ind + 1) * max_response_num - 1, max_response_num);
          }

          double out_val =
            flip_signs_beta(
              vote_v(interested_inds),
              reshape(current_param_val_v(alpha_v_lower_start_ind + interested_q_dim_inds),
                      max_response_num, interested_inds.n_elem),
              reshape(current_param_val_v(alpha_v_upper_start_ind + interested_q_dim_inds),
                      max_response_num, interested_inds.n_elem),
              current_param_val_v(leg_start_ind + j),
              reshape(current_param_val_v(delta_v_lower_start_ind + interested_q_dim_inds),
                      max_response_num, interested_inds.n_elem),
              reshape(current_param_val_v(delta_v_upper_start_ind + interested_q_dim_inds),
                      max_response_num, interested_inds.n_elem),
              question_num_choices_m1_v(question_v(interested_inds)),
              leg_mean, leg_sd, flip_beta_sd);

          if (out_val != current_param_val_v(leg_start_ind + j)) {
            flip_beta_count(j) += 1;
          }

          current_param_val_v(leg_start_ind + j) = out_val;


          }
          // vote_v(interested_inds) = out_v(
          //   span(4 * half_alpha_size, out_v.n_elem - 1));
      }

      if (pos_ind > -1 && (current_param_val_v(leg_start_ind + pos_ind) < 0)) {
        current_param_val_v(span(0, alpha_v_lower_start_ind - 1)) =
          -current_param_val_v(span(0, alpha_v_lower_start_ind - 1));
        current_param_val_v(span(delta_v_lower_start_ind, current_param_val_v.n_elem - 1)) =
          -current_param_val_v(span(delta_v_lower_start_ind, current_param_val_v.n_elem - 1));
        for (unsigned int j = 0; j < num_questions; j++) {
          for (unsigned int k = 0; k < max_response_num; k++) {
            double tmp = current_param_val_v(alpha_v_lower_start_ind + j * max_response_num + k);
            current_param_val_v(alpha_v_lower_start_ind + j * max_response_num + k) =
                -current_param_val_v(alpha_v_upper_start_ind + (j + 1) * max_response_num - 1 - k);
                current_param_val_v(alpha_v_upper_start_ind + (j + 1) * max_response_num - 1 - k) =
                -tmp;
          }
        }
      }

      if (neg_ind > -1 && pos_ind < 0 && (current_param_val_v(leg_start_ind + neg_ind) > 0)) {
        current_param_val_v(span(0, alpha_v_lower_start_ind - 1)) =
          -current_param_val_v(span(0, alpha_v_lower_start_ind - 1));
          current_param_val_v(span(delta_v_lower_start_ind, current_param_val_v.n_elem - 1)) =
          -current_param_val_v(span(delta_v_lower_start_ind, current_param_val_v.n_elem - 1));
          for (unsigned int j = 0; j < num_questions; j++) {
            for (unsigned int k = 0; k < max_response_num; k++) {
              double tmp = current_param_val_v(alpha_v_lower_start_ind + j * max_response_num + k);
              current_param_val_v(alpha_v_lower_start_ind + j * max_response_num + k) =
                -current_param_val_v(alpha_v_upper_start_ind + (j + 1) * max_response_num - 1 - k);
                current_param_val_v(alpha_v_upper_start_ind + (j + 1) * max_response_num - 1 - k) =
                -tmp;
            }
          }
      }

      int post_burn_i = i - start_iter + 1;
      if (i >= start_iter && (fmod(post_burn_i, keep_iter) == 0)) {
        int keep_iter_ind = post_burn_i / keep_iter - 1;
        all_param_draws.row(keep_iter_ind) = current_param_val_v.t();
      }
    }

    return(List::create(Named("param_draws") = all_param_draws,
                        Named("y_star_m") = y_star_m,
                        Named("swap_tracker") = flip_beta_count));
  }

// [[Rcpp::export]]
arma::mat calc_waic_ordinal_pum_utility(
    arma::vec leg_ideology,
    arma::vec alpha_m_lower, arma::vec alpha_m_upper,
    arma::vec delta_m_lower, arma::vec delta_m_upper,
    arma::mat case_vote_m, unsigned int km1) {

  mat ordinal_prob(case_vote_m.n_rows, case_vote_m.n_cols,
                   fill::ones);
  ordinal_prob = -ordinal_prob;
  for (unsigned int j = 0; j < case_vote_m.n_cols; j++) {
    for (unsigned int i = 0; i < case_vote_m.n_rows; i++) {
      if (!is_finite(case_vote_m(i, j))) {
        continue;
      }

      uvec interested_col_ind = j * km1 +
        linspace<uvec>(0, km1 - 1, km1);

      vec mean_1 =
        alpha_m_lower(interested_col_ind) % (
            leg_ideology(i) -
              delta_m_lower(interested_col_ind));
      vec mean_2 =
        alpha_m_upper(interested_col_ind) % (
          leg_ideology(i) - delta_m_upper(interested_col_ind));
      ordinal_prob(i, j) =
        calc_choice_k_prob(mean_1, mean_2, int(case_vote_m(i, j) + 0.5));
    }
  }
  return(ordinal_prob);
}

// [[Rcpp::export]]
arma::vec calc_waic_ordinal_pum_waic(
  arma::mat leg_ideology,
  arma::mat alpha_m_lower, arma::mat alpha_m_upper,
  arma::mat delta_m_lower, arma::mat delta_m_upper,
  arma::mat case_vote_m, int num_votes,
  unsigned int km1) {

  vec mean_prob(num_votes);
  mean_prob.fill(-datum::inf);
  vec mean_log_prob(num_votes, fill::zeros);
  vec log_prob_var(num_votes, fill::zeros);

  for (unsigned int iter = 0; iter < leg_ideology.n_rows; iter++) {

    Rcout << iter << endl;
    int vote_num = 0;

    for (unsigned int j = 0; j < case_vote_m.n_cols; j++) {
      for (unsigned int i = 0; i < case_vote_m.n_rows; i++) {
        if (!is_finite(case_vote_m(i, j))) {
          continue;
        }

        uvec interested_row_ind = {iter};
        uvec interested_col_ind = j * km1 +
          linspace<uvec>(0, km1 - 1, km1);

        rowvec mean_1 =
          alpha_m_lower(interested_row_ind, interested_col_ind) % (
            leg_ideology(iter, i) -
              delta_m_lower(interested_row_ind,
                            interested_col_ind));
        rowvec mean_2 =
          alpha_m_upper(interested_row_ind,
                        interested_col_ind) % (
            leg_ideology(iter, i) -
              delta_m_upper(interested_row_ind,
                            interested_col_ind));
        double yea_prob =
          calc_choice_k_prob(mean_1.t(), mean_2.t(), int(case_vote_m(i, j) + 0.5));


        yea_prob = min(yea_prob, 1 - 1e-9);
        yea_prob = max(yea_prob, 1e-9);
        double log_prob = log(yea_prob);
        mean_prob(vote_num) = log_sum_exp(mean_prob(vote_num), log_prob);
        double next_mean_log_prob = (iter * mean_log_prob(vote_num) + log_prob) / (iter + 1);
        log_prob_var(vote_num) +=
          (log_prob - mean_log_prob(vote_num)) * (log_prob - next_mean_log_prob);
        mean_log_prob(vote_num) = next_mean_log_prob;
        vote_num++;
      }
    }
  }
  return(mean_prob - log(num_votes + 0.0) -
      (log_prob_var) / (num_votes - 1.0));
}

// [[Rcpp::export]]
arma::vec calc_waic_ordinal_pum_block(
    arma::mat leg_ideology, arma::mat alpha_m_lower, arma::mat alpha_m_upper,
    arma::mat delta_m_lower, arma::mat delta_m_upper,
    arma::mat case_vote_m, arma::uvec case_year, arma::mat block_m) {

  vec mean_prob(block_m.n_rows);
  mean_prob.fill(-datum::inf);
  vec mean_log_prob(block_m.n_rows, fill::zeros);
  vec log_prob_var(block_m.n_rows, fill::zeros);

  for (unsigned int iter = 0; iter < leg_ideology.n_rows; iter++) {

    Rcout << iter << endl;

    for (int ind = 0; ind < block_m.n_rows; ind++) {
      int i = block_m(ind, 0);
      int year = block_m(ind, 1);
      double log_prob = 0;
      uvec interested_cases = find(case_year == year);
      for (unsigned int j : interested_cases) {
        if (!is_finite(case_vote_m(i, j))) {
          continue;
        }
        uvec interested_row_ind = {iter};
        uvec interested_col_ind = (j - 1) * alpha_m_lower.n_cols +
          linspace<uvec>(0, alpha_m_lower.n_cols - 1, alpha_m_lower.n_cols);
        vec mean_1 =
          alpha_m_lower(interested_row_ind, interested_col_ind) % (
              leg_ideology(iter, i) -
                delta_m_lower(interested_row_ind,
                              interested_col_ind));
        vec mean_2 =
          alpha_m_upper(interested_row_ind, interested_col_ind) % (
          leg_ideology(iter, i) -
                              delta_m_upper(interested_row_ind,
                                            interested_col_ind));
        double yea_prob =
          calc_choice_k_prob(mean_1, mean_2, int(case_vote_m(i, j) + 0.5));

        yea_prob = min(yea_prob, 1 - 1e-9);
        yea_prob = max(yea_prob, 1e-9);
        log_prob += case_vote_m(i, j) * log(yea_prob) +
          (1 - case_vote_m(i, j)) * log(1 - yea_prob);
      }
      mean_prob(ind) = max(mean_prob(ind), log_prob) +
        log(1 + exp(min(mean_prob(ind), log_prob) - max(mean_prob(ind), log_prob)));
      double next_mean_log_prob = (iter * mean_log_prob(ind) + log_prob) / (iter + 1);
      log_prob_var(ind) +=
        (log_prob - mean_log_prob(ind)) * (log_prob - next_mean_log_prob);
      mean_log_prob(ind) = next_mean_log_prob;
    }
  }
  return(
    mean_prob - log(leg_ideology.n_rows) -
      (log_prob_var) / (leg_ideology.n_rows - 1));
}

// TRUMPISSUE2_W61_alpha_lower_1 TRUMPISSUE2_W61_alpha_lower_2
//   -7.1352836                    -6.6913504
// TRUMPISSUE2_W61_alpha_lower_3 TRUMPISSUE2_W61_alpha_upper_3
//   -5.0221522                     2.5442126
// TRUMPISSUE2_W61_alpha_upper_2 TRUMPISSUE2_W61_alpha_upper_1
//   2.7547723                     6.1358179
// TRUMPISSUE2_W61_delta_lower_1 TRUMPISSUE2_W61_delta_lower_2
//   -1.4337620                    -0.7492827
// TRUMPISSUE2_W61_delta_lower_3 TRUMPISSUE2_W61_delta_upper_3
//   -0.5186346                     2.2890208
// TRUMPISSUE2_W61_delta_upper_2 TRUMPISSUE2_W61_delta_upper_1
//   2.7741386                     4.8214641

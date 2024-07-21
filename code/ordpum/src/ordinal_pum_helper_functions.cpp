#include <RcppArmadillo.h>
#include <cmath>
#include <RcppDist.h>
//Code from RcppTN: https://github.com/olmjo/RcppTN/blob/master/src/rtn1.cpp
#include "rtn1.h"

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

vec sample_alpha_ordinal_independent_lower(
    vec alpha_post_mean_m, mat alpha_post_cov_s, int num_iter = 20) {

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

double calc_choice_k_prob(
  vec mean_1, vec mean_2, int choice_k) {

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


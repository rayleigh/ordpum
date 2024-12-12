#include <RcppArmadillo.h>
#include <cmath>
#include <RcppDist.h>
//Code from RcppTN: https://github.com/olmjo/RcppTN/blob/master/src/rtn1.cpp
#include "rtn1.h"
#include <exp_cubature.h>
#include <exp_cubature_typedefs.h>
#include <mvtnormAPI.h>
#include <genzmvn.h>

//[[Rcpp::depends(RcppArmadillo, RcppDist, cubature, mvtnorm, genzmvn)]]

using namespace Rcpp;
using namespace arma;
using namespace std;
using namespace genzmvn;

const double pi2 = pow(datum::pi,2);
const double TWOPI = 6.283185307179586;


// int *n, int *nu, double *lower, double *upper,
// int *infin, double *corr, double *delta,
// int *maxpts, double *abseps, double *releps,
// double *error, double *value, int *inform, int *rnd
// extern "C" void sadmvn_(
//     int* n, double *lower, double *upper, int* infin, 
//     double* corr, int* maxpts, double *abseps, double *releps,
//     double *error, double *value, int *inform);

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
  // bool pos_ind = false, bool neg_ind = false) {
  
  // y_star_m_lower = y_star_m_lower + alpha_v_1 % delta_v_1;
  // y_star_m_upper = y_star_m_upper + alpha_v_2 % delta_v_2;
  y_star_m_lower += alpha_v_1 % delta_v_1;
  y_star_m_upper += alpha_v_2 % delta_v_2;
  // y_star_m_1 = y_star_m_1 - alpha_m.row(0) % delta_m.row(0);
  // y_star_m_2 = y_star_m_2 - alpha_m.row(1) % delta_m.row(1);
  // Rcout << "beta_fun" << endl;
  // Rcout << y_star_m_lower << endl;
  // Rcout << y_star_m_upper << endl;
  // Rcout << alpha_v_1 << endl;
  // Rcout << alpha_v_2 << endl;
  double post_var = 1.0 / pow(beta_s, 2);
  double post_mean = beta_mean / pow(beta_s, 2);
  // Rcout << post_var << endl;
  // Rcout << post_mean << endl;
  for (int j = 0; j < alpha_v_1.n_rows; j++) {
    post_var += dot(alpha_v_1.row(j), alpha_v_1.row(j)) +
      dot(alpha_v_2.row(j), alpha_v_2.row(j));
    post_mean += dot(alpha_v_1.row(j), y_star_m_lower.row(j)) + 
      dot(alpha_v_2.row(j), y_star_m_upper.row(j));
  }
  // Rcout << post_mean << endl;
  // Rcout << post_var << endl;
  return(randn() / sqrt(post_var) + post_mean / post_var);
}

// int f(unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval) {
//   double sigma = *((double *) fdata); // we can pass σ via fdata argument
//   double sum = 0;
//   unsigned i;
//   for (i = 0; i < ndim; ++i) sum += x[i] * x[i];
//   // compute the output value: note that fdim should == 1 from below
//   fval[0] = exp(-sigma * sum);
//   return 0; // success*
// }

// int truncated_gaussian_comp_v(
//     unsigned ndim, size_t npts, const double *x, void *fdata,
//     unsigned fdim, double *fval) {
//   
//   List param_info = *((List *) fdata);
//   vec mean_v = param_info(0);
//   mat cov_matrix = param_info(1);
//   bool pos_int_interval = param_info(2);
//   
//   vec tmp(x, ndim * npts);
//   mat trans_x = reshape(tmp, ndim, npts).t();
//   
//   vec change_of_v(npts);
//   change_of_v = (1 + trans_x.col(0) % trans_x.col(0)) /
//     (square(1 - trans_x.col(0) % trans_x.col(0)));
//   trans_x.col(0) = trans_x.col(0) / (1 - trans_x.col(0) % trans_x.col(0));
//   
//   if (pos_int_interval) {
//     change_of_v %=
//       (1.0 / ((1.0 - trans_x.col(1)) % (1.0 - trans_x.col(1))));
//     trans_x.col(1) = trans_x.col(1) / (1 - trans_x.col(1));
//   } else {
//     change_of_v %= 
//       (2.0 / (pow(trans_x.col(1), 3.0)));
//     trans_x.col(1) = 1.0 - 1.0 / (trans_x.col(1) % trans_x.col(1));
//   }
//   
//   vec prob_v = dmvnorm(trans_x, mean_v, cov_matrix, false) % change_of_v;
//   for (int i = 0; i < prob_v.n_elem; i++) {
//     fval[i] = prob_v(i);
//   }
//   
//   return 0;
// }

int truncated_gaussian_comp_lower(
    unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval) {
  
  List param_info = *((List *) fdata);
  vec mean_v = param_info(0);
  mat cov_matrix = param_info(1);
  bool pos_int_interval = param_info(2);
  
  vec trans_x(x, ndim);
  // trans_x = tan(trans_x);
  // double change_of_v = prod(1 / pow(cos(trans_x), 2));
  double change_of_v = 1;
  for (int i = 0; i < ndim; i++) {
    double tmp = trans_x(i);
    if (pos_int_interval) {
      trans_x(i) = tmp / (1 - tmp);
      change_of_v *= 1.0 / ((1.0 - tmp) * (1.0 - tmp));
    } else {
      trans_x(i) = 1.0 - 1.0 / (tmp * tmp);
      change_of_v *= 2.0 / (pow(tmp, 3.0));
      // trans_x(i) = tan(-datum::pi / 2 * tmp);
      // change_of_v *= 1 / (pow(cos(-datum::pi / 2 * tmp), 2.0)) * datum::pi / 2;
    }
  }
  
  double order_ok = 1;  
  for (unsigned i = 0; i < ndim - 1; ++i) {
    if (trans_x(i) > trans_x(i + 1)) {
      order_ok = 0;
      break;
    }
  }
  
  fval[0] = order_ok * as_scalar(
    dmvnorm(trans_x.t(), mean_v, cov_matrix, false)) * 
      change_of_v;
  return 0;
}

int truncated_gaussian_comp_lower_v(
    unsigned ndim, size_t npts, 
    const double *x, void *fdata, unsigned fdim, double *fval) {
  
  List param_info = *((List *) fdata);
  vec mean_v = param_info(0);
  mat cov_matrix = param_info(1);
  bool pos_int_interval = param_info(2);
  
  vec tmp(x, ndim * npts);
  mat trans_x = reshape(tmp, ndim, npts).t();
  
  vec change_of_v(npts, fill::ones);
  // vec trans_x(x, ndim);
  // trans_x = tan(trans_x);
  // double change_of_v = prod(1 / pow(cos(trans_x), 2));
  // double change_of_v = 1;
  for (int i = 0; i < ndim; i++) {
    vec tmp = trans_x.col(i);
    if (pos_int_interval) {
      trans_x.col(i) = tmp / (1 - tmp);
      change_of_v %= 1.0 / ((1.0 - tmp) % (1.0 - tmp));
    } else {
      trans_x.col(i) = 1.0 - 1.0 / (tmp % tmp);
      change_of_v %= 2.0 / (pow(tmp, 3.0));
      // trans_x(i) = tan(-datum::pi / 2 * tmp);
      // change_of_v *= 1 / (pow(cos(-datum::pi / 2 * tmp), 2.0)) * datum::pi / 2;
    }
  }

  vec prob_v = dmvnorm(trans_x, mean_v, cov_matrix, false) % change_of_v;
  for (int i = 0; i < npts; ++i) {
    double order_ok = 1;
    for (unsigned j = 0; j < ndim - 1; ++j) {
      // Rcout << j << endl;
      if (trans_x(i, j) > trans_x(i, j + 1)) {
        order_ok = 0;
        break;
      }
    }
    fval[i] = prob_v(i) * order_ok;
  }

  return 0;
}

int truncated_gaussian_comp_upper(
    unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval) {
  
  List param_info = *((List *) fdata);
  vec mean_v = param_info(0);
  mat cov_matrix = param_info(1);
  bool pos_int_interval = param_info(2);
  
  vec trans_x(x, ndim);
  // trans_x = tan(trans_x);
  // double change_of_v = prod(1 / pow(cos(trans_x), 2));
  double change_of_v = 1;
  for (int i = 0; i < ndim; i++) {
    double tmp = trans_x(i);
    if (pos_int_interval) {
      trans_x(i) = tmp / (1 - tmp);
      change_of_v *= 1.0 / ((1.0 - tmp) * (1.0 - tmp));
    } else {
      trans_x(i) = 1.0 - 1.0 / (tmp * tmp);
      change_of_v *= 2.0 / (pow(tmp, 3.0));
      // trans_x(i) = tan(-datum::pi / 2 * tmp);
      // change_of_v *= 1 / (pow(cos(-datum::pi / 2 * tmp), 2.0)) * datum::pi / 2;
    }
  }
  
  double order_ok = 1;  
  for (unsigned i = 0; i < ndim - 1; ++i) {
    if (trans_x(i) < trans_x(i + 1)) {
      order_ok = 0;
      break;
    }
  }
  
  fval[0] = order_ok * as_scalar(
    dmvnorm(trans_x.t(), mean_v, cov_matrix, false)) * 
      change_of_v;
  return 0;
}

int truncated_gaussian_comp_upper_v(
    unsigned ndim, size_t npts, 
    const double *x, void *fdata, unsigned fdim, double *fval) {
  
  List param_info = *((List *) fdata);
  vec mean_v = param_info(0);
  mat cov_matrix = param_info(1);
  bool pos_int_interval = param_info(2);
  
  vec tmp(x, ndim * npts);
  mat trans_x = reshape(tmp, ndim, npts).t();
  
  vec change_of_v(npts, fill::ones);
  // vec trans_x(x, ndim);
  // trans_x = tan(trans_x);
  // double change_of_v = prod(1 / pow(cos(trans_x), 2));
  // double change_of_v = 1;
  for (int i = 0; i < ndim; i++) {
    vec tmp = trans_x.col(i);
    if (pos_int_interval) {
      trans_x.col(i) = tmp / (1 - tmp);
      change_of_v %= 1.0 / ((1.0 - tmp) % (1.0 - tmp));
    } else {
      trans_x.col(i) = 1.0 - 1.0 / (tmp % tmp);
      change_of_v %= 2.0 / (pow(tmp, 3.0));
      // trans_x(i) = tan(-datum::pi / 2 * tmp);
      // change_of_v *= 1 / (pow(cos(-datum::pi / 2 * tmp), 2.0)) * datum::pi / 2;
    }
  }
  
  vec prob_v = dmvnorm(trans_x, mean_v, cov_matrix, false) % change_of_v;
  for (int i = 0; i < npts; ++i) {
    double order_ok = 1;
    for (unsigned j = 0; j < ndim - 1; ++j) {
      if (trans_x(i, j) < trans_x(i, j + 1)) {
        order_ok = 0;
        break;
      }
    }
    fval[i] = prob_v(i) * order_ok;
  }
  
  return 0;
}

// [[Rcpp::export]]
double sample_match_var(
  vec alpha_post_mean_v, mat alpha_post_cov_s, 
  vec delta_v, vec delta_mean_v, mat delta_cov_s) {
  
  List lower_info = List::create(
    Named("mean_v") =  alpha_post_mean_v(span(0, alpha_post_mean_v.n_elem / 2 - 1)),
    Named("cov_s") =  alpha_post_cov_s(
      span(0, alpha_post_mean_v.n_elem / 2 - 1),
      span(0, alpha_post_mean_v.n_elem / 2 - 1)),
    Named("direction") = false);
  List upper_info = List::create(
    Named("mean_v") =  alpha_post_mean_v(
      span(alpha_post_mean_v.n_elem / 2, alpha_post_mean_v.n_elem - 1)),
    Named("cov_s") =  alpha_post_cov_s(
      span(alpha_post_mean_v.n_elem / 2, alpha_post_mean_v.n_elem - 1),
      span(alpha_post_mean_v.n_elem / 2, alpha_post_mean_v.n_elem - 1)),
    Named("direction") = true);

  double sample_order_up_prob = as_scalar(dmvnorm(delta_v.t(), delta_mean_v, delta_cov_s, true));
  
  // Rcout << sample_order_up_prob << endl;
    // R::pnorm(0, post_mean(0), sqrt(1.0 / post_cov(0,0)), false, true) +
    // R::pnorm(0, post_mean(1), sqrt(1.0 / post_cov(1,1)), true, true) +
    // as_scalar(dmvnorm(delta_v.t(), delta_mean_v, delta_cov_s, false));
  
  int ret_code;
  double val, err;
  vec xmin(alpha_post_mean_v.n_elem / 2, fill::zeros);
  vec xmax(alpha_post_mean_v.n_elem / 2, fill::ones);
  // vec xmin(alpha_post_mean_v.n_elem / 2);
  // xmin.fill(-datum::pi / 2.0);
  // vec xmax(alpha_post_mean_v.n_elem / 2, fill::zeros);
  ret_code = hcubature_v(1, truncated_gaussian_comp_lower_v, &lower_info, alpha_post_mean_v.n_elem / 2, 
            xmin.begin(), xmax.begin(), 1e6, 0, 1e-4, ERROR_INDIVIDUAL, &val, &err);
  
  sample_order_up_prob += log(val);
  // val = 1;
  // Rcout << "lower" << endl;
  // Rcout << val << endl;
  // Rcout << err << endl;
  // Rcout << sample_order_up_prob << endl;

  // xmin.fill(0.0);
  // xmax.fill(datum::pi / 2.0);
  ret_code = hcubature_v(1, truncated_gaussian_comp_lower_v, &upper_info, alpha_post_mean_v.n_elem / 2, 
            xmin.begin(), xmax.begin(), 1e6, 0, 1e-4, ERROR_INDIVIDUAL, &val, &err);
  
  sample_order_up_prob += log(val);
  // val = 1;
  // Rcout << "lower" << endl;
  // Rcout << val << endl;
  // Rcout << err << endl;
  // Rcout << sample_order_up_prob << endl;
  
  double sample_order_down_prob = 
    as_scalar(dmvnorm(delta_v.t(), -delta_mean_v, delta_cov_s, true));
  // Rcout << sample_order_down_prob << endl;
  lower_info("direction") = true;
  upper_info("direction") = false;
  ret_code = hcubature_v(1, truncated_gaussian_comp_upper_v, &lower_info, alpha_post_mean_v.n_elem / 2, 
            xmin.begin(), xmax.begin(), 1e6, 0, 1e-4, ERROR_INDIVIDUAL, &val, &err);
  sample_order_down_prob += log(val);
  // val = 1;
  // Rcout << "upper" << endl;
  // Rcout << val << endl;
  // // Rcout << new_err << endl;
  // Rcout << sample_order_down_prob << endl;
  
  // xmin.fill(-datum::pi / 2.0);
  // xmax.fill(0.0);
  ret_code = hcubature_v(1, truncated_gaussian_comp_upper_v, &upper_info, alpha_post_mean_v.n_elem / 2, 
            xmin.begin(), xmax.begin(), 1e6, 0, 1e-4, ERROR_INDIVIDUAL, &val, &err);
  sample_order_down_prob += log(val);
  
  // val = 1;
  // Rcout << "upper" << endl;
  // Rcout << val << endl;
  // Rcout << sample_order_down_prob << endl;
  // 
  // Rcout << sample_order_down_prob << endl;
  
  // Rcout << "okay" << endl;
  
  // Rcout << sample_order_down_prob << endl;
  // Rcout << sample_order_up_prob << endl;
  
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

//Inspired by stackover flow comment on pmvnorm
// [[Rcpp::export]]
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

// [[Rcpp::export]]
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
  // ivec lower_int(alpha_post_mean_v.n_elem / 2, fill::zeros);
  // int* lower_int_ = lower_int.memptr();
  // ivec upper_int(alpha_post_mean_v.n_elem / 2, fill::ones);
  // int* upper_int_ = upper_int.memptr();
  
  double err;
  double val;
  int inform;
  
  // mvtnorm_C_mvtdst(&n, &nu, lower, )
  double sample_order_up_prob = as_scalar(
    dmvnorm(delta_v.t(), delta_mean_v, delta_cov_s, true));
  
  // mvtnorm_C_mvtdst(&n, &nu, zero_v_, lower_alpha_post_mean_v_,
  //                  lower_int_, lower_corr_v_, delta_, 
  //                  &maxpts, &abseps, &releps,
  //                  &err, &val, &inform, &rnd);
  sadmvn_(&n, zero_v_, lower_alpha_post_mean_v_,
          lower_int_, lower_corr_v_, 
          &maxpts, &abseps, &releps,
          &err, &val, &inform);
  sample_order_up_prob += log(val);
  // val = 1;
  // Rcout << "lower" << endl;
  // Rcout << val << endl;
  // Rcout << err << endl;
  // Rcout << sample_order_up_prob << endl;
  
  // xmin.fill(0.0);
  // xmax.fill(datum::pi / 2.0);
  // mvtnorm_C_mvtdst(&n, &nu, upper_alpha_post_mean_v_, zero_v_,
  //                  upper_int_, upper_corr_v_, delta_, 
  //                  &maxpts, &abseps, &releps,
  //                  &err, &val, &inform, &rnd);
  sadmvn_(&n, upper_alpha_post_mean_v_, zero_v_, 
          upper_int_, upper_corr_v_, 
          &maxpts, &abseps, &releps,
          &err, &val, &inform);
  sample_order_up_prob += log(val);

  double sample_order_down_prob = 
    as_scalar(dmvnorm(delta_v.t(), -delta_mean_v, delta_cov_s, true));
  // Rcout << sample_order_down_prob << endl;
  // mvtnorm_C_mvtdst(&n, &nu, lower_alpha_post_mean_v_, zero_v_, 
  //                  upper_int_, lower_corr_v_, delta_, 
  //                  &maxpts, &abseps, &releps,
  //                  &err, &val, &inform, &rnd);
  sadmvn_(&n, lower_alpha_post_mean_v_, zero_v_, 
          upper_int_, lower_corr_v_, 
          &maxpts, &abseps, &releps,
          &err, &val, &inform);
  sample_order_down_prob += log(val);
  
  // mvtnorm_C_mvtdst(&n, &nu, zero_v_, upper_alpha_post_mean_v_, 
  //                  lower_int_, upper_corr_v_, delta_, 
  //                  &maxpts, &abseps, &releps,
  //                  &err, &val, &inform, &rnd);
  sadmvn_(&n, zero_v_, upper_alpha_post_mean_v_,
          lower_int_, upper_corr_v_, 
          &maxpts, &abseps, &releps,
          &err, &val, &inform);
  sample_order_down_prob += log(val);
  
  Rcout << "Prob" << endl;
  Rcout << alpha_post_mean_v << endl;
  Rcout << sample_order_up_prob << endl;
  Rcout << sample_order_down_prob << endl;
  
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

vec sample_alpha_ordinal_interval_lower(
    vec alpha_post_mean_m, mat alpha_post_cov_s) {
  
  int Km1 = alpha_post_mean_m.n_elem / 2;
  vec out_v(alpha_post_mean_m.n_elem);
  
  mat alpha_post_cov_s_lower_chol = 
    chol(alpha_post_cov_s(span(0, Km1 - 1), 
                          span(0, Km1 - 1)), "lower");
  vec lower_trans_means = 
    solve(alpha_post_cov_s_lower_chol, 
          -alpha_post_mean_m(span(0, Km1 - 1)));
  mat alpha_post_cov_s_upper_chol = 
    chol(alpha_post_cov_s(span(Km1, out_v.n_elem - 1), 
                          span(Km1, out_v.n_elem - 1)), "lower");
  vec upper_trans_means = 
    solve(alpha_post_cov_s_upper_chol, 
          -alpha_post_mean_m(span(Km1, out_v.n_elem - 1)));
  
  for (int m = 0; m < Km1; m++) {
    out_v(m) = rtn1(0, 1, -datum::inf, lower_trans_means(m));
    out_v(m + Km1) = rtn1(0, 1, upper_trans_means(m), datum::inf);
  }
  return(join_vert(alpha_post_cov_s_lower_chol * out_v(span(0, Km1 - 1)),
                   alpha_post_cov_s_upper_chol * out_v(span(Km1, out_v.n_elem - 1))) +
            alpha_post_mean_m);  
}

vec sample_alpha_ordinal_interval_upper(
    vec alpha_post_mean_m, mat alpha_post_cov_s) {
  
  int Km1 = alpha_post_mean_m.n_elem / 2;
  vec out_v(alpha_post_mean_m.n_elem);
  
  mat alpha_post_cov_s_lower_chol = 
    chol(alpha_post_cov_s(span(0, Km1 - 1), 
                          span(0, Km1 - 1)), "lower");
  vec lower_trans_means = 
    solve(alpha_post_cov_s_lower_chol, 
          -alpha_post_mean_m(span(0, Km1 - 1)));
  mat alpha_post_cov_s_upper_chol = 
  chol(alpha_post_cov_s(span(Km1, out_v.n_elem - 1), 
                        span(Km1, out_v.n_elem - 1)), "lower");
  vec upper_trans_means = 
    solve(alpha_post_cov_s_upper_chol, 
          -alpha_post_mean_m(span(Km1, out_v.n_elem - 1)));
  
  for (int m = 0; m < Km1; m++) {
    out_v(m) = rtn1(0, 1, lower_trans_means(m), datum::inf);
    out_v(m + Km1) = rtn1(0, 1, -datum::inf, upper_trans_means(m));
  }
  return(join_vert(alpha_post_cov_s_lower_chol * out_v(span(0, Km1 - 1)),
                   alpha_post_cov_s_upper_chol * out_v(span(Km1, out_v.n_elem - 1))) +
          alpha_post_mean_m);  
}


vec sample_ordinal_probit_matched_alpha(
    mat y_star_m, vec beta_v, vec delta_v,
    vec alpha_mean_v, mat alpha_cov_s,
    vec delta_mean_v, mat delta_cov_s) {
  
  // vec diff_sq_v = {
  //   dot(beta_diff_v_1, beta_diff_v_1),
  //   dot(beta_diff_v_2, beta_diff_v_2)};
  
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
  
  vec output_v(post_mean.n_elem + 1);
  output_v(post_mean.n_elem) = sample_match_var(
    post_mean, post_cov, delta_v, delta_mean_v, delta_cov_s);
  
  // Rcout << "okay" << endl;
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

// [[Rcpp::export]]
vec sample_ordinal_probit_matched_alpha_intervals(
    mat y_star_m, vec beta_v, vec delta_v,
    vec alpha_mean_v, mat alpha_cov_s,
    vec delta_mean_v, mat delta_cov_s) {
  
  // vec diff_sq_v = {
  //   dot(beta_diff_v_1, beta_diff_v_1),
  //   dot(beta_diff_v_2, beta_diff_v_2)};
  
  // Rcout << "Stop 1" << endl;
  
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
  
  // Rcout << post_mean << endl;
  // Rcout << post_cov << endl;
  
  // Rcout << "Stop 2" << endl;
  
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
  
  // Rcout << int_post_mean << endl;
  // Rcout << int_post_cov << endl;
  
  // Rcout << int_post_cov << endl;
  // Rcout << det(int_post_cov) << endl;
  // Rcout << "Stop 2" << endl;
  
  vec output_v(post_mean.n_elem + 1);
  output_v(post_mean.n_elem) = sample_match_var_mvtnorm(
    int_post_mean, int_post_cov, delta_v, delta_mean_v, delta_cov_s);
  
  // Rcout << "Stop 3" << endl;
  
  // Rcout << "okay" << endl;
  // mat ones(alpha_mean_v.n_elem / 2, alpha_mean_v.n_elem / 2, fill::ones);
  
  // if (output_v(post_mean.n_elem) > 0) {
  //   output_v(span(0, post_mean.n_elem - 1)) =
  //     solve(trans_m, sample_alpha_ordinal_interval_lower(
  //             post_mean, post_cov));
  // } else {
  //   output_v(span(0, post_mean.n_elem - 1)) =
  //     solve(trans_m, sample_alpha_ordinal_interval_upper(
  //             post_mean, post_cov));
  // }
  
  if (output_v(post_mean.n_elem) > 0) {
    output_v(span(0, post_mean.n_elem - 1)) =
      sample_alpha_ordinal_independent_lower(
        post_mean, post_cov);
  } else {
    output_v(span(0, post_mean.n_elem - 1)) =
      sample_alpha_ordinal_independent_upper(
        post_mean, post_cov);
  }
  
  // Rcout << "Stop 4" << endl;
  
  return(output_v);
}

// [[Rcpp::export]]
vec sample_ordinal_utility_matched_delta(
    mat y_star_m_lower, mat y_star_m_upper, 
    vec alpha_v_lower, vec alpha_v_upper, 
    vec beta_v, double match_var,
    vec delta_mean_v, mat delta_cov_s) {
  
  y_star_m_lower -= resize(beta_v * alpha_v_lower.t(), size(y_star_m_lower));
  y_star_m_upper -= resize(beta_v * alpha_v_upper.t(), size(y_star_m_upper));
  y_star_m_lower *= -1;
  y_star_m_upper *= -1;
  // y_star_m_1 = y_star_m_1 + alpha_v(0) * beta_v.t(); 
  // y_star_m_2 = y_star_m_2 + alpha_v(1) * beta_v.t(); 
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


// int divonne_fWrapper(const int *nDim, const double x[],
//                      const int *nComp, double f[], void *userdata, const int *nVec,
//                      const int *core, const int *phase) {
//   
//   Rcpp::NumericVector xVal = Rcpp::NumericVector(x, x + (*nDim) * (*nVec));  /* The x argument for the R function f */
//   ii_ptr iip = (ii_ptr) userdata;
//   if (iip -> vector_intf) {
//     // Make the argument vector appear as a matrix for R
//     xVal.attr("dim") = Rcpp::Dimension(*nDim, *nVec);
//   }
//   
//   Rcpp::NumericVector fx;
//   
//   if (iip -> cuba_args) {
//     Rcpp::IntegerVector phaseVal = Rcpp::IntegerVector(phase, phase + 1);  /* The phase argument for the R function f */        
//     fx = Rcpp::Function(iip -> fun)(xVal, Rcpp::_["cuba_phase"] = phaseVal);
//   } else {
//     fx = Rcpp::Function(iip -> fun)(xVal);
//   }
//   double* fxp = fx.begin();         /* The ptr to f(x) (real) vector */
//   for (int i = 0; i < (*nComp) * (*nVec); ++i) {
//     f[i] = fxp[i];
//   }
//   return 0;
// }

int truncated_gaussian_alpha_gamma_comp_lower(
    const int *nDim, const double x[],
    const int *nComp, double f[], void *userdata, const int *nVec,
    const int *core, const int *phase) {
  
  List param_info = *((List *) userdata);
  vec mean_v = param_info(0);
  mat cov_matrix = param_info(1);
  bool pos_int_interval = param_info(2);
  
  vec trans_x(x, *nDim);
  // trans_x = tan(trans_x);
  // double change_of_v = prod(1 / pow(cos(trans_x), 2));
  double change_of_v = 1;
  for (int i = 0; i < *nDim / 2; i++) {
    double tmp = trans_x(i);
    trans_x(i) = tmp / (1.0 - tmp * tmp);
    change_of_v *= (1 + tmp * tmp) / 
      (pow(1 - tmp * tmp, 2));
  }
  
  for (int i = *nDim / 2; i < *nDim; i++) {
    double tmp = trans_x(i);
    if (pos_int_interval) {
      trans_x(i) = tmp / (1 - tmp);
      change_of_v *= 1.0 / ((1.0 - tmp) * (1.0 - tmp));
    } else {
      trans_x(i) = 1.0 - 1.0 / (tmp * tmp);
      change_of_v *= 2.0 / (pow(tmp, 3.0));
      // trans_x(i) = tan(-datum::pi / 2 * tmp);
      // change_of_v *= 1 / (pow(cos(-datum::pi / 2 * tmp), 2.0)) * datum::pi / 2;
    }
  }
  
  double order_ok = 1;  
  for (unsigned i = *nDim / 2; i < *nDim - 1; ++i) {
    if (trans_x(i) > trans_x(i + 1)) {
      order_ok = 0;
      break;
    }
  }
  
  f[0] = order_ok * as_scalar(
    dmvnorm(trans_x.t(), mean_v, cov_matrix, false)) * 
      change_of_v;
  return 0;
}

int truncated_gaussian_alpha_gamma_comp_upper(
    const int *nDim, const double x[],
    const int *nComp, double f[], void *userdata, const int *nVec,
    const int *core, const int *phase) {
  
  List param_info = *((List *) userdata);
  vec mean_v = param_info(0);
  mat cov_matrix = param_info(1);
  bool pos_int_interval = param_info(2);
  
  vec trans_x(x, *nDim);
  // trans_x = tan(trans_x);
  // double change_of_v = prod(1 / pow(cos(trans_x), 2));
  double change_of_v = 1;
  for (int i = 0; i < *nDim / 2; i++) {
    double tmp = trans_x(i);
    trans_x(i) = tmp / (1.0 - tmp * tmp);
    change_of_v *= (1 + tmp * tmp) / 
      (pow(1 - tmp * tmp, 2));
  }
  
  for (int i = *nDim / 2; i < *nDim; i++) {
    double tmp = trans_x(i);
    if (pos_int_interval) {
      trans_x(i) = tmp / (1 - tmp);
      change_of_v *= 1.0 / ((1.0 - tmp) * (1.0 - tmp));
    } else {
      trans_x(i) = 1.0 - 1.0 / (tmp * tmp);
      change_of_v *= 2.0 / (pow(tmp, 3.0));
      // trans_x(i) = tan(-datum::pi / 2 * tmp);
      // change_of_v *= 1 / (pow(cos(-datum::pi / 2 * tmp), 2.0)) * datum::pi / 2;
    }
  }
  
  double order_ok = 1;  
  for (unsigned i = *nDim / 2; i < *nDim / 2 - 1; ++i) {
    if (trans_x(i) < trans_x(i + 1)) {
      order_ok = 0;
      break;
    }
  }
  
  f[0] = order_ok * as_scalar(
    dmvnorm(trans_x.t(), mean_v, cov_matrix, false)) * 
      change_of_v;
  return 0;
}

// double sample_match_var_alpha_gamma(
//     vec alpha_post_mean_v_up,
//     vec alpha_post_mean_v_down,
//     vec gamma_post_mean_v_up,
//     vec gamma_post_mean_v_down,
//     mat alpha_gamma_post_cov_s) {
// 
//   int Km1 = gamma_post_mean_v_up.n_elem;
// 
//   uvec lower_inds = join_vert(
//     linspace<uvec>(0, Km1 / 2 - 1, Km1 / 2),
//     linspace<uvec>(Km1, Km1 + Km1 / 2 - 1,
//                    Km1 / 2));
//   uvec upper_inds = join_vert(
//     linspace<uvec>(Km1 / 2, Km1 - 1,Km1 / 2),
//     linspace<uvec>(Km1 + Km1 / 2,
//                    2 * Km1 - 1, Km1 / 2));
//   vec mean_v = join_vert(
//     gamma_post_mean_v_up(span(0, Km1 / 2 - 1)),
//     alpha_post_mean_v_up(span(0, Km1 / 2 - 1)));
// 
//   List lower_info = List::create(
//     Named("mean_v") =
//       (vec) join_vert(
//         gamma_post_mean_v_up(span(0, Km1 / 2 - 1)),
//         alpha_post_mean_v_up(span(0, Km1 / 2 - 1))),
//     Named("cov_s") =  (mat) alpha_gamma_post_cov_s.submat(lower_inds, lower_inds),
//     Named("direction") = false);
// 
//   // List lower_info;
//   List upper_info = List::create(
//     Named("mean_v") = (vec) join_vert(
//       gamma_post_mean_v_up(span(Km1 / 2, Km1 - 1)),
//       alpha_post_mean_v_up(span(Km1 / 2, Km1 - 1))),
//     Named("cov_s") = (mat) alpha_gamma_post_cov_s.submat(upper_inds, upper_inds),
//     Named("direction") = true);
// 
//   // Divonne(const int ndim, const int ncomp,
//   //         integrand_t integrand, void *userdata, const int nvec,
//   //         const cubareal epsrel, const cubareal epsabs,
//   //         const int flags, const int seed,
//   //         const int mineval, const int maxeval,
//   //         const int key1, const int key2, const int key3, const int maxpass,
//   //         const cubareal border, const cubareal maxchisq, const cubareal mindeviation,
//   //         const int ngiven, const int ldxgiven, cubareal xgiven[],
//   //         const int nextra, peakfinder_t peakfinder,
//   //         const char *statefile, void *spin,
//   //         int *nregions, int *neval, int *fail,
//   //         cubareal integral[], cubareal error[], cubareal prob[]);
// 
//   // double sample_order_up_prob = as_scalar(dmvnorm(delta_v.t(), delta_mean_v, delta_cov_s, true));
//   // Rcout << sample_order_up_prob << endl;
//   // R::pnorm(0, post_mean(0), sqrt(1.0 / post_cov(0,0)), false, true) +
//   // R::pnorm(0, post_mean(1), sqrt(1.0 / post_cov(1,1)), true, true) +
//   // as_scalar(dmvnorm(delta_v.t(), delta_mean_v, delta_cov_s, false));
// 
//   int nregions, fail, num_iter;
//   double val, err, prob;
//   vec xmin = join_vert(
//     -ones(Km1),
//     zeros(Km1));
//   vec xmax(2 * Km1, fill::ones);
//   // vec xmin(alpha_post_mean_v.n_elem / 2);
//   // xmin.fill(-datum::pi / 2.0);
//   // vec xmax(alpha_post_mean_v.n_elem / 2, fill::zeros);
//   cubacores(0, 0);
//   num_iter = 0;
//   // Divonne(
//   //   xmin.n_elem, 1, (integrand_t) truncated_gaussian_comp_lower,
//   //   (void *) &lower_info, 1, 1e-5, 1e-12, 4, 0, 0, 1e6,
//   //   47, 1, 1, 5, 0, 10, 0.25, 0, xmin.n_elem, NULL, 0,
//   //   NULL, NULL, NULL, &nregions, &num_iter, &fail,
//   //   &val, &err, &prob);
//   double sample_order_up_prob = log(val);
//   // val = 1;
//   // Rcout << "lower" << endl;
//   Rcout << val << endl;
//   // Rcout << err << endl;
//   // Rcout << sample_order_up_prob << endl;
// 
//   // xmin.fill(0.0);
//   // xmax.fill(datum::pi / 2.0);
//   cubacores(0, 0);
//   num_iter = 0;
//   // Divonne(xmin.n_elem, 1, (integrand_t) truncated_gaussian_comp_lower,
//   //         (void *) &upper_info, 1, 1e-5, 1e-12, 4, 0, 0, 1e6,
//   //         47, 1, 1, 5, 0, 10, 0.25, 0, xmin.n_elem, NULL, 0,
//   //         NULL, NULL, NULL, &nregions, &num_iter, &fail,
//   //         &val, &err, &prob);
//   sample_order_up_prob += log(val);
//   // val = 1;
//   // Rcout << "lower" << endl;
//   Rcout << val << endl;
//   // Rcout << err << endl;
//   // Rcout << sample_order_up_prob << endl;
// 
//   // Rcout << sample_order_down_prob << endl;
//   lower_info("mean_v") =
//     (vec) join_vert(
//       gamma_post_mean_v_down(span(0, Km1 / 2 - 1)),
//       alpha_post_mean_v_down(span(0, Km1 / 2 - 1)));
//   lower_info("direction") = true;
//   upper_info("mean_v") =
//     (vec) join_vert(
//         gamma_post_mean_v_up(span(Km1 / 2, Km1 - 1)),
//         alpha_post_mean_v_up(span(Km1 / 2, Km1 - 1)));
//   upper_info("direction") = false;
//   cubacores(0, 0);
//   num_iter = 0;
//   // Divonne(xmin.n_elem, 1, (integrand_t) truncated_gaussian_comp_upper,
//   //         (void *) &lower_info, 1, 1e-5, 1e-12, 4, 0, 0, 1e6,
//   //         47, 1, 1, 5, 0, 10, 0.25, 0, xmin.n_elem, NULL, 0,
//   //         NULL, NULL, NULL, &nregions, &num_iter, &fail,
//   //         &val, &err, &prob);
//   double sample_order_down_prob = log(val);
//   // val = 1;
//   // Rcout << "upper" << endl;
//   // Rcout << val << endl;
//   // // Rcout << new_err << endl;
//   // Rcout << sample_order_down_prob << endl;
// 
//   // xmin.fill(-datum::pi / 2.0);
//   // xmax.fill(0.0);
//   cubacores(0, 0);
//   num_iter = 0;
//   // Divonne(xmin.n_elem, 1, (integrand_t) truncated_gaussian_comp_upper,
//   //         (void *) &upper_info, 1, 1e-5, 1e-12, 4, 0, 0, 1e6,
//   //         47, 1, 1, 5, 0, 10, 0.25, 0, xmin.n_elem, NULL, 0,
//   //         NULL, NULL, NULL, &nregions, &num_iter, &fail,
//   //         &val, &err, &prob);
//   sample_order_down_prob += log(val);
//   // val = 1;
//   // Rcout << "upper" << endl;
//   // Rcout << val << endl;
//   // Rcout << sample_order_down_prob << endl;
//   //
//   // Rcout << sample_order_down_prob << endl;
// 
//   // Rcout << "okay" << endl;
// 
//   // Rcout << sample_order_down_prob << endl;
//   // Rcout << sample_order_up_prob << endl;
// 
//   double match_var;
// 
//   if (!is_finite(sample_order_down_prob) &&
//       !is_finite(sample_order_up_prob)) {
//       sample_order_up_prob = 0;
//     sample_order_down_prob = 0;
//   }
//   double log_sample_prob = sample_order_up_prob -
//     (max(sample_order_up_prob, sample_order_down_prob) +
//     log(1 + exp(min(sample_order_up_prob, sample_order_down_prob) -
//     max(sample_order_up_prob, sample_order_down_prob))));
//   match_var = (log(randu()) < log_sample_prob) * 2 - 1;
//   return(match_var);
// }

vec sample_alpha_gamma_ordinal_independent_lower_gamma(
    vec out_v, vec alpha_post_mean_v, vec gamma_post_mean_v, 
    mat post_cov_s) {
  
  mat alpha_post_cov_s = post_cov_s(
    span(gamma_post_mean_v.n_elem, out_v.n_elem - 1),
    span(gamma_post_mean_v.n_elem, out_v.n_elem - 1));
  
  vec gamma_cond_mean = gamma_post_mean_v + 
    post_cov_s(
      span(0, gamma_post_mean_v.n_elem - 1),
      span(gamma_post_mean_v.n_elem, out_v.n_elem - 1)) *
        solve(alpha_post_cov_s,
              (out_v(span(gamma_post_mean_v.n_elem, out_v.n_elem - 1)) - alpha_post_mean_v));
  mat gamma_cond_cov_m = 
    post_cov_s(
      span(0, gamma_post_mean_v.n_elem - 1),
      span(0, gamma_post_mean_v.n_elem - 1)) +
    post_cov_s(
      span(0, gamma_post_mean_v.n_elem - 1),
      span(gamma_post_mean_v.n_elem, out_v.n_elem - 1)) * 
        solve(alpha_post_cov_s,
              post_cov_s(
                span(gamma_post_mean_v.n_elem, out_v.n_elem - 1),
                span(0, gamma_post_mean_v.n_elem - 1)));
  
  out_v(span(0, gamma_post_mean_v.n_elem / 2 - 1)) = 
    rmvnorm(1, gamma_cond_mean(span(0, gamma_post_mean_v.n_elem / 2 - 1)),
            gamma_cond_cov_m(span(0, gamma_post_mean_v.n_elem / 2 - 1),
                             span(0, gamma_post_mean_v.n_elem / 2 - 1)));
  
  out_v(span(gamma_post_mean_v.n_elem / 2, gamma_post_mean_v.n_elem - 1)) = 
    rmvnorm(1, gamma_cond_mean(span(gamma_post_mean_v.n_elem / 2, gamma_post_mean_v.n_elem - 1)),
            gamma_cond_cov_m(span(gamma_post_mean_v.n_elem / 2, gamma_post_mean_v.n_elem - 1),
                             span(gamma_post_mean_v.n_elem / 2, gamma_post_mean_v.n_elem - 1)));
  return(out_v);
}


vec sample_alpha_gamma_ordinal_independent_lower_alpha(
    vec alpha_post_mean_v, vec gamma_post_mean_v, 
    mat post_cov_s) {
  
  vec out_v(alpha_post_mean_v.n_elem + gamma_post_mean_v.n_elem);
  mat alpha_post_cov_s = post_cov_s(
    span(gamma_post_mean_v.n_elem, out_v.n_elem - 1),
    span(gamma_post_mean_v.n_elem, out_v.n_elem - 1));
  
  out_v(gamma_post_mean_v.n_elem) = 
    rtn1(alpha_post_mean_v(0), sqrt(alpha_post_cov_s(0, 0)), 
        -datum::inf, 0);
  for (int m = 1; m < alpha_post_mean_v.n_elem / 2; m++) {
    out_v(gamma_post_mean_v.n_elem + m) = 
      rtn1(alpha_post_mean_v(m), sqrt(alpha_post_cov_s(m, m)), 
            out_v(m - 1), 0);
  } 
  
  out_v(alpha_post_mean_v.n_elem / 2) =
    rtn1(alpha_post_mean_v(alpha_post_mean_v.n_elem / 2), 
         sqrt(alpha_post_cov_s(alpha_post_mean_v.n_elem / 2, 
              alpha_post_mean_v.n_elem / 2)), 
         0, datum::inf);
  for (int m = alpha_post_mean_v.n_elem / 2 + 1; m < alpha_post_mean_v.n_elem; m++) {
    out_v(gamma_post_mean_v.n_elem + m) = 
      rtn1(alpha_post_mean_v(m), sqrt(alpha_post_cov_s(m, m)), 
          out_v(m - 1), datum::inf);
  } 
  
  return(out_v);
}

vec sample_alpha_gamma_ordinal_independent_upper_alpha(
    vec alpha_post_mean_v, vec gamma_post_mean_v, 
    mat post_cov_s) {
  
  vec out_v(alpha_post_mean_v.n_elem + gamma_post_mean_v.n_elem);
  mat alpha_post_cov_s = post_cov_s(
    span(gamma_post_mean_v.n_elem, out_v.n_elem - 1),
    span(gamma_post_mean_v.n_elem, out_v.n_elem - 1));
  
  out_v(out_v.n_elem - 1) = 
    rtn1(alpha_post_mean_v(alpha_post_mean_v.n_elem - 1), 
         sqrt(alpha_post_cov_s(alpha_post_mean_v.n_elem - 1, 
                               alpha_post_mean_v.n_elem - 1)), 
         -datum::inf, 0);
  for (int m = alpha_post_mean_v.n_elem - 2; 
       m > alpha_post_mean_v.n_elem / 2 - 1; m--) {
    out_v(m + gamma_post_mean_v.n_elem) = 
      rtn1(alpha_post_mean_v(m), sqrt(alpha_post_cov_s(m, m)), 
          out_v(m + 1), 0);
  } 
  
  out_v(gamma_post_mean_v.n_elem + 
          alpha_post_mean_v.n_elem / 2 - 1) =
    rtn1(alpha_post_mean_v(alpha_post_mean_v.n_elem / 2 - 1), 
         sqrt(alpha_post_cov_s(alpha_post_mean_v.n_elem / 2 - 1, 
                               alpha_post_mean_v.n_elem / 2 - 1)), 
         0, datum::inf);
  for (int m = alpha_post_mean_v.n_elem / 2 - 2; m > -1; m--) {
    out_v(gamma_post_mean_v.n_elem + m) = 
      rtn1(alpha_post_mean_v(m), sqrt(alpha_post_cov_s(m, m)), 
            out_v(m + 1), datum::inf);
  } 
  return(out_v);
}

vec sample_ordinal_pum_alpha_gamma(
  mat y_star_m_lower, mat y_star_m_upper, 
  vec beta_v, vec alpha_mean_v, vec alpha_cov_s,
  vec gamma_mean_v, vec gamma_cov_s) {
  
  vec post_mean_up = join_vert(
    solve(gamma_cov_s, gamma_mean_v),
    solve(alpha_cov_s, alpha_mean_v));
  vec post_mean_down = join_vert(
    solve(gamma_cov_s, -gamma_mean_v),
    solve(alpha_cov_s, alpha_mean_v));
  mat post_cov_s(gamma_mean_v.n_elem + alpha_mean_v.n_elem,
                 gamma_mean_v.n_elem + alpha_mean_v.n_elem,
                 fill::zeros);
  post_cov_s(span(0, gamma_mean_v.n_elem - 1),
             span(0, gamma_mean_v.n_elem - 1)) = gamma_cov_s.i();
  post_cov_s(span(gamma_mean_v.n_elem, 
                  post_cov_s.n_rows - 1),
             span(gamma_mean_v.n_elem,
                  post_cov_s.n_rows - 1)) = alpha_cov_s.i();
  
  vec post_mean(post_mean_up.n_elem, fill::zeros);
  for (int i = 0; i < beta_v.n_elem; i++) {
    mat beta_i = join_horiz(
      -eye(alpha_mean_v.n_elem, alpha_mean_v.n_elem),
      beta_v(i) * eye(alpha_mean_v.n_elem, alpha_mean_v.n_elem));
    post_cov_s += beta_i.t() * beta_i;
    post_mean += beta_i.t() * 
      join_vert(y_star_m_lower.row(i).t(),
                y_star_m_upper.row(i).t());
  }
  post_mean_up = solve(post_cov_s, post_mean_up + post_mean);
  post_mean_down = solve(post_cov_s, post_mean_down + post_mean);
  post_cov_s = post_cov_s.i();
  
  // double match_var = sample_match_var_alpha_gamma(
  //   post_mean_up(span(gamma_mean_v.n_elem, post_mean.n_elem - 1)),
  //   post_mean_down(span(gamma_mean_v.n_elem, post_mean.n_elem - 1)),
  //   post_mean_up(span(0, gamma_mean_v.n_elem - 1)),
  //   post_mean_down(span(0, gamma_mean_v.n_elem - 1)),
  //   post_cov_s);
  double match_var = 0;
  
  vec out_v(post_mean.n_elem);
  if (match_var > 0) {
    out_v = sample_alpha_gamma_ordinal_independent_lower_alpha(
      post_mean_up(span(gamma_mean_v.n_elem, post_mean.n_elem - 1)),
      post_mean_up(span(0, gamma_mean_v.n_elem - 1)),
      post_cov_s);
    out_v = sample_alpha_gamma_ordinal_independent_lower_gamma(
      out_v, post_mean_up(span(gamma_mean_v.n_elem, post_mean.n_elem - 1)),
      post_mean_up(span(0, gamma_mean_v.n_elem - 1)),
      post_cov_s);
  } else {
    out_v = sample_alpha_gamma_ordinal_independent_upper_alpha(
      post_mean_down(span(gamma_mean_v.n_elem, post_mean.n_elem - 1)),
      post_mean_down(span(0, gamma_mean_v.n_elem - 1)),
      post_cov_s);
    out_v = sample_alpha_gamma_ordinal_independent_lower_gamma(
      out_v, post_mean_down(span(gamma_mean_v.n_elem, post_mean.n_elem - 1)),
      post_mean_down(span(0, gamma_mean_v.n_elem - 1)),
      post_cov_s);
  }
  return(out_v);
}

// [[Rcpp::export]]
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
  
  // Rcout << "okay" << endl;
  
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
    // mvtnorm_C_mvtdst(&n, &nu, post_mean_v_, zero_v_, 
    //                  upper_int_, corr_v_, delta_, 
    //                  &maxpts, &abseps, &releps,
    //                  &err, &val, &inform, &rnd);
    sadmvn_(&n, post_mean_v_, zero_v_, upper_int_, corr_v_,
            &maxpts, &abseps, &releps, &err, &val, &inform);
    yea_prob = val;
    // mvtnorm_C_mvtdst(&n, &nu, post_mean_v_2_, zero_v_, 
    //                  upper_int_, corr_v_, delta_, 
    //                  &maxpts, &abseps, &releps,
    //                  &err, &val, &inform, &rnd);
    sadmvn_(&n, post_mean_v_2_, zero_v_, upper_int_, corr_v_,
            &maxpts, &abseps, &releps, &err, &val, &inform);                 
    yea_prob += val;                  
  } else {
    for (int i = 0; i < mean_1.n_elem; i++) {
      post_mean_v_[i] = mean_1(i) / sqrt(2.0);
      post_mean_v_[i + mean_1.n_elem] = mean_2(i) / sqrt(2.0);
    }
    // mvtnorm_C_mvtdst(&n, &nu, post_mean_v_, zero_v_, 
    //                  upper_int_, corr_v_, delta_, 
    //                  &maxpts, &abseps, &releps,
    //                  &err, &val, &inform, &rnd);
    sadmvn_(&n, post_mean_v_, zero_v_, upper_int_, corr_v_,
            &maxpts, &abseps, &releps, &err, &val, &inform);
    yea_prob = val;
                     
  }
  return(yea_prob);
}

vec flip_signs(
    ivec vote, vec alpha_v_lower, vec alpha_v_upper, vec beta_v, 
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
  // y_star(K) = rtn1(0, 1, -datum::inf,
  //   max(y_star(k), y_star(2 * K - 2 - k)));
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
  // y_star(K) = rtn1(0, 1, -datum::inf,
  //   max(y_star(k), y_star(2 * K - 2 - k)));
}

// [[Rcpp::export]]
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
  uvec vote_v, uvec respondent_v, uvec question_v, mat all_param_draws, mat y_star_m,
  int leg_start_ind, int alpha_v_lower_start_ind, int alpha_v_upper_start_ind,
  int delta_v_lower_start_ind, int delta_v_upper_start_ind,
  double leg_mean, double leg_sd, vec alpha_mean_v, mat alpha_cov_s,
  vec delta_mean_v, mat delta_cov_s, 
  int num_iter, int start_iter, int keep_iter, int pos_ind, int neg_ind) {

  vec current_param_val_v = all_param_draws.row(0).t();
  int half_alpha_size = (y_star_m.n_cols - 1) / 2;
  int num_questions = (alpha_v_upper_start_ind - alpha_v_lower_start_ind) / half_alpha_size;
  int num_ind = alpha_v_lower_start_ind - leg_start_ind;
  
  // vec accept_count(zeta_param_start_ind - psi_param_start_ind);
  // accept_count.zeros();
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


  // sample_ordinal_utility_probit_beta(
  //   mat y_star_m_lower, mat y_star_m_upper, 
  //   mat alpha_v_1, mat alpha_v_2,
  //   mat delta_v_1, mat delta_v_2,
  //   double beta_mean, double beta_s)
    for (unsigned int j = 0; j < num_ind; j++) {
      uvec interested_inds = find(respondent_v == j);
      uvec interested_q_dim_inds(interested_inds.n_elem * half_alpha_size);
      for (int k = 0; k < interested_inds.n_elem; k++) {
        int question_ind = question_v(interested_inds(k));
        // Rcout << question_ind << endl;
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

    // sample_ordinal_probit_matched_alpha(
    //   mat y_star_m, vec beta_v, vec delta_v,
    //   vec alpha_mean_v, mat alpha_cov_s,
    //   vec delta_mean_v, mat delta_cov_s)
  
    
    vec match_var_v(num_questions);
    for (unsigned int j = 0; j < num_questions; j++) {
      // Rcout << j << endl;
      uvec interested_inds = find(question_v == j);
      uvec interested_q_inds =
        linspace<uvec>(j * half_alpha_size,
                       (j + 1) * half_alpha_size - 1, half_alpha_size);
      // Rcout << y_star_m_1.submat(interested_inds, current_ind) << endl;
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
    // Rcout << match_var_v << endl;
    
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
// 
// vec adjust_all_judge_ideology(
//     vec current_param_val_v, 
//     uvec judge_start_ind,
//     uvec case_year_v, uvec case_judge_year_v,
//     int alpha_v_1_start_ind, int alpha_v_2_start_ind,
//     int delta_v_1_start_ind, int delta_v_2_start_ind,
//     uvec pos_judge_ind, uvec pos_judge_year,
//     uvec neg_judge_ind, uvec neg_judge_year) {
//   
//   
//   for (int i = 0; i < pos_judge_ind.n_elem; i++) {
//     if (current_param_val_v(pos_judge_ind(i)) < 0) {
//       uvec judge_year = find(case_judge_year_v == pos_judge_year(i));
//       uvec cases = find(case_year_v == pos_judge_year(i));
//       current_param_val_v(judge_year) =
//         -current_param_val_v(judge_year);
//       current_param_val_v(alpha_v_1_start_ind + cases) = 
//         -current_param_val_v(alpha_v_1_start_ind + cases);
//       current_param_val_v(alpha_v_2_start_ind + cases) = 
//         -current_param_val_v(alpha_v_2_start_ind + cases);
//       current_param_val_v(delta_v_1_start_ind + cases) = 
//         -current_param_val_v(delta_v_1_start_ind + cases);
//       current_param_val_v(delta_v_2_start_ind + cases) = 
//         -current_param_val_v(delta_v_2_start_ind + cases);
//     }
//   }
//   for (int i = 0; i < neg_judge_ind.n_elem; i++) {
//     if (current_param_val_v(neg_judge_ind(i)) > 0) {
//       uvec judge_year = find(case_judge_year_v == neg_judge_year(i));
//       uvec cases = find(case_year_v == neg_judge_year(i));
//       current_param_val_v(judge_year) =
//         -current_param_val_v(judge_year);
//       current_param_val_v(alpha_v_1_start_ind + cases) = 
//         -current_param_val_v(alpha_v_1_start_ind + cases);
//       current_param_val_v(alpha_v_2_start_ind + cases) = 
//         -current_param_val_v(alpha_v_2_start_ind + cases);
//       current_param_val_v(delta_v_1_start_ind + cases) = 
//         -current_param_val_v(delta_v_1_start_ind + cases);
//       current_param_val_v(delta_v_2_start_ind + cases) = 
//         -current_param_val_v(delta_v_2_start_ind + cases);
//     }
//   }
//   return(current_param_val_v);
// }
// 
// List sample_three_utility_probit_gp(
//     mat vote_m, mat all_param_draws, mat y_star_m_1, mat y_star_m_2, mat y_star_m_3,
//     uvec judge_start_inds, uvec judge_end_inds, uvec case_years, 
//     umat case_judge_years_ind_m, uvec judge_year_v,
//     int alpha_v_1_start_ind, int alpha_v_2_start_ind, 
//     int delta_v_1_start_ind, int delta_v_2_start_ind, int rho_ind,
//     vec alpha_mean_v, mat alpha_cov_s, vec delta_mean_v, mat delta_cov_s, 
//     double rho_mean,double rho_sigma, double rho_sd, int num_iter, int start_iter, 
//     int keep_iter, uvec pos_judge_ind, uvec neg_judge_ind,
//     uvec pos_judge_year, uvec neg_judge_year) {
//   
//   
//   vec current_param_val_v = all_param_draws.row(0).t();
//   // vec accept_count(zeta_param_start_ind - psi_param_start_ind);
//   // accept_count.zeros();
//   for (int i = 0; i < num_iter; i++) {
//     if (i % 100 == 0) {
//       Rcout << i << "\n";
//     }
//   
//     for (int j = 0; j < vote_m.n_rows; j++) {
//       for (int k = 0; k < vote_m.n_cols; k++) {
//         if (!is_finite(vote_m(j, k))) {
//           continue;
//         }
//         vec y_star_vec = {y_star_m_1(j, k), 
//                           y_star_m_2(j, k), 
//                           y_star_m_3(j, k)};
//         vec out_v = sample_y_star_m(
//           y_star_vec, vote_m(j, k), 
//           current_param_val_v(alpha_v_1_start_ind + k),
//           current_param_val_v(alpha_v_2_start_ind + k),
//           current_param_val_v(judge_start_inds(j) + case_judge_years_ind_m(j, k)), 
//           current_param_val_v(delta_v_1_start_ind + k), 
//           current_param_val_v(delta_v_2_start_ind + k));
//         y_star_m_1(j, k) = out_v(0);  
//         y_star_m_2(j, k) = out_v(1);
//         y_star_m_3(j, k) = out_v(2);
//       }
//     }
//     
//     for (unsigned int j = 0; j < vote_m.n_rows; j++) {
//       uvec current_ind = {j};
//       uvec interested_inds = find_finite(vote_m.row(j).t());
//       rowvec y_star_m_1_v = y_star_m_1.row(j);
//       rowvec y_star_m_3_v = y_star_m_3.row(j);
//       uvec judge_years_v = case_judge_years_ind_m.row(j).t();
//       current_param_val_v(span(
//           judge_start_inds(j), judge_end_inds(j))) =
//         sample_three_utility_probit_beta_gp(
//           y_star_m_1.submat(current_ind, interested_inds),
//           y_star_m_3.submat(current_ind, interested_inds),
//           current_param_val_v(alpha_v_1_start_ind + interested_inds).t(),
//           current_param_val_v(alpha_v_2_start_ind + interested_inds).t(),
//           current_param_val_v(delta_v_1_start_ind + interested_inds).t(),
//           current_param_val_v(delta_v_2_start_ind + interested_inds).t(),
//           judge_years_v(interested_inds), current_param_val_v(rho_ind));
//     }
//     
//     vec match_var_v(vote_m.n_cols);
//     for (unsigned int j = 0; j < vote_m.n_cols; j++) {
//       uvec current_ind = {j};
//       uvec interested_inds = find_finite(vote_m.col(j));
//       vec delta_v = {current_param_val_v(delta_v_1_start_ind + j),
//                      current_param_val_v(delta_v_2_start_ind + j)};
//       uvec judge_years_v = case_judge_years_ind_m.col(j);
//       vec out_v =
//         sample_three_utility_probit_matched_alpha(
//           y_star_m_1.submat(interested_inds, current_ind), 
//           y_star_m_3.submat(interested_inds, current_ind),  
//           current_param_val_v(
//             judge_start_inds(interested_inds) + 
//             judge_years_v(interested_inds)), 
//           delta_v, alpha_mean_v, alpha_cov_s,
//           delta_mean_v, delta_cov_s); 
//       
//       current_param_val_v(alpha_v_1_start_ind + j) = out_v(0);
//       current_param_val_v(alpha_v_2_start_ind + j) = out_v(1);
//       match_var_v(j) = out_v(2);
//     }
//     
//     for (unsigned int j = 0; j < vote_m.n_cols; j++) {
//       uvec current_ind = {j};
//       uvec interested_inds = find_finite(vote_m.col(j));
//       vec alpha_v = {current_param_val_v(alpha_v_1_start_ind + j),
//                      current_param_val_v(alpha_v_2_start_ind + j)};
//       uvec judge_years_v = case_judge_years_ind_m.col(j);
//       vec out_v =
//         sample_three_utility_probit_matched_delta(
//           y_star_m_1.submat(interested_inds, current_ind), 
//           y_star_m_3.submat(interested_inds, current_ind),
//           alpha_v, current_param_val_v(
//               judge_start_inds(interested_inds) + 
//                 judge_years_v(interested_inds)), 
//           match_var_v(j), delta_mean_v, delta_cov_s); 
//       current_param_val_v(delta_v_1_start_ind + j) = out_v(0);
//       current_param_val_v(delta_v_2_start_ind + j) = out_v(1);
//     }
//     
//     if (pos_judge_ind.n_elem > 0 || neg_judge_ind.n_elem > 0) {
//       current_param_val_v(span(0, rho_ind - 1)) =
//         adjust_all_judge_ideology(
//           current_param_val_v(span(0, rho_ind - 1)), 
//           judge_start_inds, case_years, judge_year_v,
//           alpha_v_1_start_ind, alpha_v_2_start_ind,
//           delta_v_1_start_ind, delta_v_2_start_ind,
//           pos_judge_ind, pos_judge_year,
//           neg_judge_ind, neg_judge_year);
//     }
//     
//     current_param_val_v(rho_ind) = sample_rho_pos_logit_gibbs(
//       current_param_val_v(rho_ind), 
//       current_param_val_v(span(0, alpha_v_1_start_ind - 1)), 
//       judge_start_inds, judge_end_inds, rho_mean, rho_sigma, rho_sd);
//     
//     int post_burn_i = i - start_iter + 1;
//     if (i >= start_iter && (fmod(post_burn_i, keep_iter) == 0)) {
//       int keep_iter_ind = post_burn_i / keep_iter - 1;
//       all_param_draws.row(keep_iter_ind) = current_param_val_v.t();
//     }
//   }
//   
//   return(List::create(Named("param_draws") = all_param_draws, 
//                       Named("y_star_m_1") = y_star_m_1, 
//                       Named("y_star_m_2") = y_star_m_2, 
//                       Named("y_star_m_3") = y_star_m_3));
// }


double phid(double x) {
  return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

// BVND calculates the probability that X > DH and Y > DK.
// Note: Prob( X < DH, Y < DK ) = BVND( -DH, -DK, R )
// Code and description is adopted from tvpack.f in the 
// mvtnorm package with help from ChatGPT
// [[Rcpp::export]]
double bvnd(double DH, double DK, double R) {
  
  vec x;
  vec w;
  // double as = 0.0;
  // double a = 0.0;
  double b = 0.0;
  // double c = 0.0;
  // double d = 0.0;
  double rs = 0.0;
  double xs = 0.0;
  double bvn = 0.0;
  // double sn = 0.0;
  // double asr = 0.0;
  double h = DH;
  double k = DK;
  double hk = h * k;
  
  if (std::abs(R) < 0.3) {
    x = {-0.9324695142031522,-0.6612093864662647,
         -0.2386191860831970};
    w = {0.1713244923791705, 
         0.3607615730481384, 0.4679139345726904};
  } else if (std::abs(R) < 0.75) {
    x = {-0.9815606342467191, -0.9041172563704750,
         -0.7699026741943050, -0.5873179542866171,
         -0.3678314989981802, -0.1252334085114692};
    w = {0.4717533638651177e-01, 0.1069393259953183, 
         0.1600783285433464, 0.2031674267230659,
         0.2334925365383547, 0.2491470458134029};
  } else {
    x = {-0.9931285991850949, -0.9639719272779138,
         -0.9122344282513259, -0.8391169718222188,
         -0.7463319064601508, -0.6360536807265150,
         -0.5108670019508271, -0.3737060887154196,
         -0.2277858511416451, -0.7652652113349733e-01};
    w = {0.1761400713915212e-01, 0.4060142980038694e-01, 
         0.6267204833410906e-01, 0.8327674157670475e-01,
         0.1019301198172404, 0.1181945319615184,
         0.1316886384491766, 0.1420961093183821,
         0.1491729864726037, 0.1527533871307259};
  }

  if (std::abs(R) < 0.925) {
    if (std::abs(R) > 0.0) {
      double hs = (h * h + k * k) / 2.0;
      double asr = std::asin(R);
      for (int i = 0; i < x.n_elem; ++i) {
        for (int is = -1; is <= 1; is += 2) {
          double sn = std::sin(asr * (is * x[i] + 1) / 2.0);
          bvn += w[i] * std::exp((sn * hk - hs) / (1.0 - sn * sn));
        }
      }
      bvn = bvn * asr / (2.0 * TWOPI);
    }
    bvn += R::pnorm(-h, 0, 1,-datum::inf, false) * R::pnorm(-k, 0, 1,-datum::inf, false);
  } else {
    if (R < 0.0) {
      k = -k;
      hk = -hk;
    }
    if (std::abs(R) < 1.0) {
      double as = (1.0 - R) * (1.0 + R);
      double a = std::sqrt(as);
      double bs = std::pow(h - k, 2);
      double c = (4.0 - hk) / 8.0;
      double d = (12.0 - hk) / 16.0;
      double asr = -(bs / as + hk) / 2.0;
      if (asr > -100.0) {
        bvn = a * std::exp(asr) * 
          (1.0 - c * (bs - as) * (1.0 - d * bs / 5.0) / 3.0 + 
          c * d * as * as / 5);
      }
      if (-hk < 100) {
        b = sqrt(bs);
        bvn = bvn - exp(-hk/2) * sqrt(TWOPI) * R::pnorm(-b/a, 0, 1,-datum::inf, false) * b
                * (1 - c * bs * (1 - d * bs/5) / 3);
      }
      a = a / 2;
      for (int i = 0; i < x.n_elem; i++) {
        for (int is = -1; is <= 1; is += 2) {
          xs = pow(a * (is * x[i] + 1), 2);
          rs = sqrt(1 - xs);
          asr = -(bs/xs + hk) / 2;
          if (asr > -100) {
            bvn = bvn + a * w[i] * exp(asr)
                   * (exp(-hk * (1 - rs) / (2 * (1 + rs)))/rs
                        - (1 + c * xs * (1 + d * xs)));
            }
          }
        }
        bvn = -bvn/TWOPI;
    }
    if (R > 0) {
      bvn = bvn + R::pnorm(-std::max(h, k), 0, 1,-datum::inf, false);
    } else {
      bvn = -bvn;
      if (k > h) {
        bvn = bvn + R::pnorm(k, 0, 1,-datum::inf, false) - R::pnorm(h, 0, 1,-datum::inf, false);
      }
    }
  }
  return(bvn);
}

// [[Rcpp::export]]
mat calc_probit_bggum_three_utility_post_prob_m(
    mat leg_ideology, mat alpha_m, mat delta_m,
    mat case_vote_m, int num_votes) {
  
  mat post_prob(case_vote_m.n_rows, case_vote_m.n_cols, fill::zeros);
  for (int iter = 0; iter < leg_ideology.n_rows; iter++) {
    for (int j = 0; j < case_vote_m.n_cols; j++) {
      for (int i = 0; i < case_vote_m.n_rows; i++) {
        double mean_1 = 
          alpha_m(iter, 2 * j) * (
              leg_ideology(iter, i) - delta_m(iter, 2 * j));
        double mean_2 = 
          alpha_m(iter, 2 * j + 1) * (
              leg_ideology(iter, i) - delta_m(iter, 2 * j + 1));
        post_prob(i, j) += bvnd(-mean_1 / sqrt(2), -mean_2 / sqrt(2), 0.5);
      }
    }
  }
  return(post_prob);
}

// [[Rcpp::export]]
mat calc_waic_ordinal_pum_utility(
    vec leg_ideology, 
    vec alpha_m_lower, vec alpha_m_upper, 
    vec delta_m_lower, vec delta_m_upper,
    mat case_vote_m, unsigned int km1) {
  
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
vec calc_waic_ordinal_pum_waic(
  mat leg_ideology, 
  mat alpha_m_lower, mat alpha_m_upper, 
  mat delta_m_lower, mat delta_m_upper,
  mat case_vote_m, int num_votes, 
  unsigned int km1) {
  
  vec mean_prob(num_votes);
  mean_prob.fill(-datum::inf);
  vec mean_log_prob(num_votes, fill::zeros);
  vec log_prob_var(num_votes, fill::zeros);
  
  // int num_position = 
  //   alpha_m_lower.n_cols / (case_vote_m.n_cols);

  // Rcout << case_vote_m << endl;
  // double corr = 0.5;
  // double sd = sqrt(2);
  // mat lower_cov = {{2, 1},
  //                  {1, 2}};
  for (unsigned int iter = 0; iter < leg_ideology.n_rows; iter++) {
    // if (iter + 1 % 100 == 0) {
    //   Rcout << iter << "\n";
    // }
    Rcout << iter << endl;
    int vote_num = 0;
    // Rcout << vote_num << endl;
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
        
          // bvnd(-mean_1 / sqrt(2), -mean_2 / sqrt(2), 0.5);
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
    // Rcout << vote_num << endl;
  }
  return(mean_prob - log(num_votes + 0.0) -
      (log_prob_var) / (num_votes - 1.0));
}

// [[Rcpp::export]]
vec calc_waic_ordinal_pum_block(
    mat leg_ideology, mat alpha_m_lower, mat alpha_m_upper, 
    mat delta_m_lower, mat delta_m_upper,
    mat case_vote_m, uvec case_year, mat block_m) {
  
  vec mean_prob(block_m.n_rows);
  mean_prob.fill(-datum::inf);
  vec mean_log_prob(block_m.n_rows, fill::zeros);
  vec log_prob_var(block_m.n_rows, fill::zeros);
  // Rcout << case_vote_m << endl;
  // double corr = 0.5;
  // double sd = sqrt(2);
  // mat lower_cov = {{2, 1},
  //                  {1, 2}};
  for (unsigned int iter = 0; iter < leg_ideology.n_rows; iter++) {
    // if (iter + 1 % 100 == 0) {
    //   Rcout << iter << "\n";
    // }
    Rcout << iter << endl;
    // int vote_num = 0;
    // Rcout << vote_num << endl;
    for (int ind = 0; ind < block_m.n_rows; ind++) {
      int i = block_m(ind, 0);
      int year = block_m(ind, 1);
      // int judge_ind = i + (year - 1) * case_vote_m.n_rows;
      double log_prob = 0;
      uvec interested_cases = find(case_year == year);
      // Rcout << interested_cases << endl;
      for (unsigned int j : interested_cases) {
        if (!is_finite(case_vote_m(i, j))) {
          continue;
        }
        // int judge_ind = i + (case_year(j) - 1) * case_vote_m.n_rows;
        // Rcout << judge_ind << endl;
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
    // Rcout << vote_num << endl;
  }
  return(
    mean_prob - log(leg_ideology.n_rows) -
      (log_prob_var) / (leg_ideology.n_rows - 1));
}

// [[Rcpp::export]]
vec calc_waic_probit_bggum_three_utility_block_rcpp(
    mat leg_ideology, mat alpha_m, mat delta_m,
    mat case_vote_m, uvec case_year, mat block_m) {
  
  vec mean_prob(block_m.n_rows);
  mean_prob.fill(-datum::inf);
  vec mean_log_prob(block_m.n_rows, fill::zeros);
  vec log_prob_var(block_m.n_rows, fill::zeros);
  // Rcout << case_vote_m << endl;
  // double corr = 0.5;
  // double sd = sqrt(2);
  // mat lower_cov = {{2, 1},
  //                  {1, 2}};
  for (int iter = 0; iter < leg_ideology.n_rows; iter++) {
    // if (iter + 1 % 100 == 0) {
    //   Rcout << iter << "\n";
    // }
    Rcout << iter << endl;
    // int vote_num = 0;
    // Rcout << vote_num << endl;
    for (int ind = 0; ind < block_m.n_rows; ind++) {
      int i = block_m(ind, 0);
      int year = block_m(ind, 1);
      // int judge_ind = i + (year - 1) * case_vote_m.n_rows;
      double log_prob = 0;
      uvec interested_cases = find(case_year == year);
      for (int j : interested_cases) {
        if (!is_finite(case_vote_m(i, j))) {
          continue;
        }
        // int judge_ind = i + (case_year(j) - 1) * case_vote_m.n_rows;
        // Rcout << judge_ind << endl;
        double mean_1 = 
          alpha_m(iter, 2 * j) * (
              leg_ideology(iter, ind) - delta_m(iter, 2 * j));
        double mean_2 = 
          alpha_m(iter, 2 * j + 1) * (
              leg_ideology(iter, ind) - delta_m(iter, 2 * j + 1));
        double yea_prob = 
          bvnd(-mean_1 / sqrt(2), -mean_2 / sqrt(2), 0.5);
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
    // Rcout << vote_num << endl;
  }
  return(
    mean_prob - log(leg_ideology.n_rows) -
      (log_prob_var) / (leg_ideology.n_rows - 1));
}

// [[Rcpp::export]]
vec calc_waic_probit_bggum_three_utility_block_vote_rcpp(
    mat leg_ideology, mat alpha_m, mat delta_m,
    mat case_vote_m, mat block_m) {
  
  vec mean_prob(block_m.n_rows);
  mean_prob.fill(-datum::inf);
  vec mean_log_prob(block_m.n_rows, fill::zeros);
  vec log_prob_var(block_m.n_rows, fill::zeros);
  // Rcout << case_vote_m << endl;
  // double corr = 0.5;
  // double sd = sqrt(2);
  // mat lower_cov = {{2, 1},
  //                  {1, 2}};
  for (int iter = 0; iter < leg_ideology.n_rows; iter++) {
    // if (iter + 1 % 100 == 0) {
    //   Rcout << iter << "\n";
    // }
    Rcout << iter << endl;
    // int vote_num = 0;
    // Rcout << vote_num << endl;
    for (int ind = 0; ind < block_m.n_rows; ind++) {
      int j = block_m(ind, 0);
      // int year = block_m(ind, 1);
      // int judge_ind = i + (year - 1) * case_vote_m.n_rows;
      double log_prob = 0;
      for (int i = 0; i < case_vote_m.n_rows; i++) {
        if (!is_finite(case_vote_m(i, j))) {
          continue;
        }
        // int judge_ind = i + (case_year(j) - 1) * case_vote_m.n_rows;
        // Rcout << judge_ind << endl;
        double mean_1 = 
          alpha_m(iter, 2 * j) * (
              leg_ideology(iter, i) - delta_m(iter, 2 * j));
        double mean_2 = 
          alpha_m(iter, 2 * j + 1) * (
              leg_ideology(iter, i) - delta_m(iter, 2 * j + 1));
        double yea_prob = 
          // mvtnorm_C_mvtdst(&n, &nu, zero_v_, upper_alpha_post_mean_v_, 
          //                  lower_int_, upper_corr_v_, delta_, 
          //                  &maxpts, &abseps, &releps,
          //                  &err, &val, &inform, &rnd);
          
          bvnd(-mean_1 / sqrt(2), -mean_2 / sqrt(2), 0.5);
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
    // Rcout << vote_num << endl;
  }
  return(
    mean_prob - log(leg_ideology.n_rows) -
      (log_prob_var) / (leg_ideology.n_rows - 1));
}


// double err;
// double val;
// int inform;
// 
// // mvtnorm_C_mvtdst(&n, &nu, lower, )
// double sample_order_up_prob = as_scalar(
//   dmvnorm(delta_v.t(), delta_mean_v, delta_cov_s, true));
// 
// mvtnorm_C_mvtdst(&n, &nu, zero_v_, lower_alpha_post_mean_v_,
//                  lower_int_, lower_corr_v_, delta_, 
//                  &maxpts, &abseps, &releps,
//                  &err, &val, &inform, &rnd);
//                  sample_order_up_prob += log(val);
//                  // val = 1;
//                  // Rcout << "lower" << endl;
//                  // Rcout << val << endl;
//                  // Rcout << err << endl;
//                  // Rcout << sample_order_up_prob << endl;
//                  
//                  // xmin.fill(0.0);
//                  // xmax.fill(datum::pi / 2.0);
//                  mvtnorm_C_mvtdst(&n, &nu, upper_alpha_post_mean_v_, zero_v_,
//                                   upper_int_, upper_corr_v_, delta_, 
//                                   &maxpts, &abseps, &releps,
//                                   &err, &val, &inform, &rnd);
//                                   sample_order_up_prob += log(val);

// vec sample_y_star_m_no(vec y_star_no, double mean_m_1, double mean_m_2) {
//   
//   mat lower_cov = {{2, 1},
//                    {1, 2}};
//   mat upper_cov = {{2, 1},
//                    {1, 2}};
//   upper_cov = upper_cov * upper_cov.t();
//   double mean_diff = mean_m_1 - mean_m_2;
//   
//   // lower_cov <- matrix(c(1, -1, 0, -1, 0, 1), byrow = T, nrow = 2)
//   // lower_cov <- lower_cov %*% t(lower_cov)
//   // upper_cov <- matrix(c(0, -1, 1, -1, 0, 1), byrow = T, nrow = 2)
//   // upper_cov <- upper_cov %*% t(upper_cov)
//   // mean_diff_m <- mean_m_1 - mean_m_2
//   
//   double lower_prob = pmvnorm_rcpp(
//     {-mean_m_1, -mean_diff},
//     {0, 0}, lower_cov);
//   double upper_prob = pmvnorm_rcpp(
//     {-mean_m_2, mean_diff},
//     {0, 0}, upper_cov);
//   double prob = upper_prob / (lower_prob + upper_prob);
//   prob = max(prob, 1e-9);
//   prob = min(prob, 1 - 1e-9);
//   if (randu() < prob) {
//     y_star_no(0) = 
//       r_truncnorm(-mean_m_1, 1, -datum::inf, y_star_no(2));
//     y_star_no(1) = 
//       r_truncnorm(0, 1, -datum::inf, y_star_no(2));
//     y_star_no(2) = 
//       r_truncnorm(-mean_m_2, 1, max(y_star_no(0), y_star_no(1)), datum::inf);
//   } else {
//     y_star_no(0) = 
//       r_truncnorm(-mean_m_1, 1, max(y_star_no(1), y_star_no(2)), datum::inf);
//     y_star_no(1) = 
//       r_truncnorm(0, 1, -datum::inf, y_star_no(0));
//     y_star_no(2) = 
//       r_truncnorm(-mean_m_2, 1, -datum::inf, y_star_no(0));
//   }
//   return(y_star_no);
// }
// 
// mat sample_y_star_m_three_utility_three_all(
//     mat y_star_m, imat vote_m, vec alpha_m_1, vec alpha_m_2, 
//     vec delta_m_1, vec delta_m_2, vec beta_v, 
//     mat three_utility_cov_m) {
//     
//     // int pos_counts = 0;
//     // int neg_counts = 0;
//     // int na_counts = 0;
//     for (int i = 0; i < vote_m.n_rows; i++) {
//       for (int j = 0; j < vote_m.n_cols; j++) {
//         double mean_m_1 = alpha_m_1(j) * (beta_v(i) - delta_m_1(j));
//         double mean_m_2 = alpha_m_2(j) * (beta_v(i) - delta_m_2(j));
//         int vote_ind = j * vote_m.n_rows + i;
//         if (vote_m(i, j) == 1) {
//           y_star_m.col(vote_ind) = sample_y_star_m_yea(
//             y_star_m.col(vote_ind), mean_m_1, mean_m_2);
//           // pos_counts++;
//         } else if (vote_m(i, j) == 0) {
//           // neg_counts++;
//           y_star_m.col(vote_ind) = sample_y_star_m_no(
//             y_star_m.col(vote_ind), mean_m_1, mean_m_2);
//         } else {
//           // if (!is_finite(vote_m(i, j))) {
//             // na_counts++;
//             // Rcout << vote_m(i, j) << "\n";
//             y_star_m.col(vote_ind) = sample_y_star_m_na(mean_m_1, mean_m_2);
//           // } 
//         }
//           
//       }
//     }
//     // Rcout << pos_counts << "\n";
//     // Rcout << neg_counts << "\n";
//     // Rcout << na_counts << "\n";
//     return(y_star_m);
// }

// mat sample_three_utility_probit(
//     mat all_param_draws, imat vote_m, mat y_star_m,
//     int beta_start_ind, int alpha_start_ind,
//     int delta_start_ind, mat three_utility_cov_m_three,
//     mat three_utility_cov_m_two,
//     double leg_mean, double leg_s,
//     vec alpha_mean, mat alpha_s,
//     vec delta_mean, mat delta_s,
//     int num_iter, int start_iter, int keep_iter,
//     int pos_ind, int neg_ind) {
//   
//   
//   vec current_param_val_v = all_param_draws.row(0).t();
//   for (int i = 0; i < num_iter; i++) {
//     y_star_m = sample_y_star_m_three_utility_three_all( 
//       y_star_m, vote_m, 
//       current_param_val_v(span(alpha_start_ind, 
//                                alpha_start_ind + vote_m.n_cols - 1)),
//       current_param_val_v(span(alpha_start_ind + vote_m.n_cols, 
//                                alpha_start_ind + 2 * vote_m.n_cols - 1)),
//       current_param_val_v(span(delta_start_ind, 
//                                delta_start_ind + vote_m.n_cols - 1)),
//       current_param_val_v(span(delta_start_ind + vote_m.n_cols, 
//                                delta_start_ind + 2 * vote_m.n_cols - 1)),                              
//       current_param_val_v(span(beta_start_ind, 
//                                beta_start_ind + vote_m.n_rows - 1)),
//       three_utility_cov_m_three);    
//     
//     Rcout << "okay" << "\n";
//     {
//       mat alpha_m = current_param_val_v(span(alpha_start_ind + vote_m.n_cols, 
//                                              alpha_start_ind + 2 * vote_m.n_cols - 1));
//       alpha_m.reshape(vote_m.n_cols, 2);
//       alpha_m = alpha_m.t();
//       mat delta_m = current_param_val_v(span(delta_start_ind + vote_m.n_cols, 
//                                              delta_start_ind + 2 * vote_m.n_cols - 1));
//       delta_m.reshape(vote_m.n_cols, 2);
//       delta_m = delta_m.t();
//       for (int j = 0; j < vote_m.n_rows; j++) {
//           // mat y_star_m, mat alpha_m, mat delta_m, mat three_utility_cov_m,
//           // double beta_mean, double beta_s, 
//           // bool pos_ind = false, bool neg_ind = false
//           uvec interested_cols = linspace<uvec>(j, (vote_m.n_cols - 1) * vote_m.n_rows + j, 
//                                                 vote_m.n_cols);
//           uvec interested_rows = {0, 2};
//           current_param_val_v(beta_start_ind + j) =
//             sample_three_utility_probit_beta(
//               y_star_m(interested_rows, interested_cols), 
//               alpha_m, delta_m,
//               three_utility_cov_m_two, leg_mean, leg_s,
//               pos_ind == j, neg_ind == j);
//       }
//     }
//     
//     for (int j = 0; j < vote_m.n_cols; j++) {
//       uvec interested_cols = linspace<uvec>(j * vote_m.n_rows, (j + 1) * vote_m.n_rows - 1, 
//                                             vote_m.n_rows);
//       uvec interested_rows = {0, 2};
//       // vec alpha_v, mat y_star_m, 
//       // vec beta_v, vec delta_v, mat three_utility_cov_m,
//       // vec alpha_mean_v, mat alpha_cov_s
//       vec out_v =
//         sample_three_utility_probit_delta(
//           y_star_m(interested_rows, interested_cols),
//           {current_param_val_v(alpha_start_ind + j),
//            current_param_val_v(alpha_start_ind + vote_m.n_cols + j)},
//            current_param_val_v(span(beta_start_ind, beta_start_ind + vote_m.n_rows - 1)),
//            three_utility_cov_m_two, delta_mean, delta_s);
//       current_param_val_v(delta_start_ind + j) = out_v(0);
//       current_param_val_v(delta_start_ind + vote_m.n_cols + j) = out_v(1);
//     }
//     Rcout << "okay" << "\n";
//     
//     for (int j = 0; j < vote_m.n_cols; j++) {
//       uvec interested_cols = linspace<uvec>(j * vote_m.n_rows, (j + 1) * vote_m.n_rows - 1, 
//                                             vote_m.n_rows);
//       uvec interested_rows = {0, 2};
//       // vec alpha_v, mat y_star_m, 
//       // vec beta_v, vec delta_v, mat three_utility_cov_m,
//       // vec alpha_mean_v, mat alpha_cov_s
//       // vec alpha_v = {current_param_val_v(alpha_start_ind + j),
//       //                current_param_val_v(alpha_start_ind + vote_m.n_cols + j)};
//       // vec delta_v = {current_param_val_v(delta_start_ind + j),
//       //                current_param_val_v(delta_start_ind + vote_m.n_cols + j)};
//       vec out_v =
//         sample_three_utility_probit_alpha(
//         {current_param_val_v(alpha_start_ind + j),
//          current_param_val_v(alpha_start_ind + vote_m.n_cols + j)},
//           y_star_m(interested_rows, interested_cols),
//           current_param_val_v(span(beta_start_ind, beta_start_ind + vote_m.n_rows - 1)),
//           {current_param_val_v(delta_start_ind + j),
//            current_param_val_v(delta_start_ind + vote_m.n_cols + j)}, 
//         three_utility_cov_m_two, alpha_mean, alpha_s);
//       if (!out_v.is_finite()) {
//         Rcout << out_v << "\n";  
//       }
//       current_param_val_v(alpha_start_ind + j) = out_v(0);
//       current_param_val_v(alpha_start_ind + vote_m.n_cols + j) = out_v(1);
//     }
//     Rcout << "okay" << "\n";
//     int post_burn_i = i - start_iter + 1;
//     if (i >= start_iter && (fmod(post_burn_i, keep_iter) == 0)) {
//       int keep_iter_ind = post_burn_i / keep_iter - 1;
//       all_param_draws.row(keep_iter_ind) = current_param_val_v.t();
//     }
//   }
//   return(all_param_draws);
// }

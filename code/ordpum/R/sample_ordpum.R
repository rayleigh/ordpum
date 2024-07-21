#' ordpum: A package for sampling from the ordinal probit unfolding model
#' @name ordpum
#' @useDynLib ordpum, .registration=TRUE
#'
#' @importFrom stats rbinom
init_y_star_m <- function(response_v, K) {

  y_star_m = matrix(0, nrow = length(response_v), ncol = 2 * K - 1)
  y_star_m[response_v == K, K] = 1
  for (i in which(response_v != K)) {
    upper = rbinom(1, 1, 0.5)
    if (upper > 0) {
      y_star_m[i, response_v[i]] = 1
    } else {
      y_star_m[i, 2 * K - response_v[i]] = 1
    }
  }
  return(y_star_m)
}

init_y_star_m_gen_choices <- function(response_v, num_choices_v, K) {

  y_star_m = matrix(0, nrow = length(response_v), ncol = 2 * K - 1)
  for (i in 1:length(response_v)) {
    if (response_v[i] == num_choices_v[i]) {
      y_star_m[i, K] = 1
      next
    }
    upper = rbinom(1, 1, 0.5)
    if (upper > 0) {
      y_star_m[i, response_v[i]] = 1
    } else {
      y_star_m[i, 2 * K - response_v[i]] = 1
    }
  }
  return(y_star_m)
}

init_data_rcpp <- function(vote_m, K, leg_pos_init = NULL,
                           alpha_pos_init = NULL, delta_pos_init = NULL,
                           y_star_m_init = NULL, total_iter) {

  data_m <- expand.grid(1:nrow(vote_m), 1:ncol(vote_m))
  data_m$vote <- as.vector(vote_m)
  data_m <- data_m[!is.na(data_m[,3]),]

  if (!is.null(leg_pos_init)) {
    leg_pos_m <-
      matrix(leg_pos_init, nrow = total_iter, ncol = nrow(vote_m), byrow = T)
  } else {
    leg_pos_m <- matrix(0, nrow = total_iter, ncol = nrow(vote_m))
  }

  if (!is.null(alpha_pos_init)) {
    alpha_pos_m <-
      matrix(t(alpha_pos_init), nrow = total_iter, ncol = 2 * (K - 1) * ncol(vote_m), byrow = T)
  } else {
    alpha_pos_m <-
      matrix(c(rep((K:2 - 1) * -0.5, ncol(vote_m)), rep((2:K - 1) * 0.5, ncol(vote_m))),
             nrow = total_iter, ncol = 2 * (K - 1) * ncol(vote_m), byrow = T)
  }

  if (!is.null(delta_pos_init)) {
    delta_pos_m <-
      matrix(t(delta_pos_init), nrow = total_iter, ncol = 2 * (K - 1) * ncol(vote_m), byrow = T)
  } else {
    delta_pos_m <-
      matrix(0, nrow = total_iter, ncol = 2 * (K - 1) * ncol(vote_m), byrow = T)
  }

  if (!is.null(y_star_m_init)) {
    y_star_m <- y_star_m_init
  } else {
    y_star_m <- init_y_star_m(data_m[,3], K)
  }

  all_params_draw <- cbind(leg_pos_m, alpha_pos_m, delta_pos_m)
  beta_start_ind = 0;
  alpha_lower_start_ind = nrow(vote_m);
  alpha_upper_start_ind = alpha_lower_start_ind + ncol(alpha_pos_m) / 2;
  delta_lower_start_ind = alpha_upper_start_ind + ncol(alpha_pos_m) / 2;
  delta_upper_start_ind = delta_lower_start_ind + ncol(delta_pos_m) / 2;

  return(list(data_m - 1, all_params_draw, y_star_m,
              beta_start_ind, alpha_lower_start_ind, alpha_upper_start_ind,
              delta_lower_start_ind, delta_upper_start_ind))
}

init_data_rcpp_gen_choices <- function(vote_m, K, num_choices,
                                       leg_pos_init = NULL,
                                       alpha_pos_init = NULL, delta_pos_init = NULL,
                                       y_star_m_init = NULL, total_iter) {

  data_m <- expand.grid(1:nrow(vote_m), 1:ncol(vote_m))
  data_m$vote <- as.vector(vote_m)
  data_m$choices_m1 <- as.vector(num_choices)
  data_m <- data_m[!is.na(data_m[,3]),]

  if (!is.null(leg_pos_init)) {
    leg_pos_m <-
      matrix(leg_pos_init, nrow = total_iter, ncol = nrow(vote_m), byrow = T)
  } else {
    leg_pos_m <- matrix(0, nrow = total_iter, ncol = nrow(vote_m))
  }

  if (!is.null(alpha_pos_init)) {
    alpha_pos_m <-
      matrix(t(alpha_pos_init), nrow = total_iter, ncol = 2 * (K - 1) * ncol(vote_m), byrow = T)
  } else {
    alpha_init <- sapply(1:ncol(num_choices), function(i) {
      alpha_lower <- rep(0, K - 1)
      alpha_lower[1:(num_choices[1, i] - 1)] <-
        (num_choices[1, i]:2 - 1) * -0.5
      alpha_upper <- rep(0, K - 1)
      alpha_upper[K - 1 - (num_choices[1, i]:2 - 2)] <-
        (2:num_choices[1, i] - 1) * 0.5
      c(alpha_lower, alpha_upper)
    })
    alpha_init <- cbind(alpha_init[1:(K - 1),], alpha_init[1:(K - 1) + (K - 1),])
    alpha_pos_m <-
      matrix(alpha_init, nrow = total_iter, ncol = 2 * (K - 1) * ncol(vote_m), byrow = T)
  }

  if (!is.null(delta_pos_init)) {
    delta_pos_m <-
      matrix(t(delta_pos_init), nrow = total_iter, ncol = 2 * (K - 1) * ncol(vote_m), byrow = T)
  } else {
    delta_pos_m <-
      matrix(0, nrow = total_iter, ncol = 2 * (K - 1) * ncol(vote_m), byrow = T)
  }

  if (!is.null(y_star_m_init)) {
    y_star_m <- y_star_m_init
  } else {
    y_star_m <- init_y_star_m_gen_choices(data_m[,3], data_m[,4], K)
  }

  all_params_draw <- cbind(leg_pos_m, alpha_pos_m, delta_pos_m)
  beta_start_ind = 0;
  alpha_lower_start_ind = nrow(vote_m);
  alpha_upper_start_ind = alpha_lower_start_ind + ncol(alpha_pos_m) / 2;
  delta_lower_start_ind = alpha_upper_start_ind + ncol(alpha_pos_m) / 2;
  delta_upper_start_ind = delta_lower_start_ind + ncol(delta_pos_m) / 2;

  return(list(data_m - 1, all_params_draw, y_star_m,
              beta_start_ind, alpha_lower_start_ind, alpha_upper_start_ind,
              delta_lower_start_ind, delta_upper_start_ind))
}

#' @importFrom Rcpp evalCpp
#' @export sample_ordinal_utility_probit_rcpp
sample_ordinal_utility_probit_rcpp <- function(
  vote_m, K, leg_mean, leg_s, alpha_mean, alpha_cov_s,
  delta_mean, delta_cov_s, num_iter = 2000, start_iter = 0, keep_iter = 1,
  leg_pos_init = NULL, alpha_pos_init = NULL, delta_pos_init = NULL,
  y_star_m = NULL, pos_ind = 0, neg_ind = 0, start_val = NULL) {

  total_iter = (num_iter - start_iter) %/% keep_iter
  init_info <- init_data_rcpp(
    vote_m, K, leg_pos_init, alpha_pos_init, delta_pos_init,
    y_star_m, total_iter)

  if (!is.null(start_val)) {
    init_info[[2]][1,] <- start_val
  }

  draw_info <- sample_ordinal_utility_probit(
    init_info[[1]][,3], init_info[[1]][,1], init_info[[1]][,2],
    init_info[[2]], init_info[[3]], init_info[[4]],
    init_info[[5]], init_info[[6]],
    init_info[[7]], init_info[[8]], leg_mean, leg_s,
    alpha_mean, alpha_cov_s, delta_mean, delta_cov_s,
    num_iter, start_iter, keep_iter, pos_ind - 1, neg_ind - 1)

  all_param_draw = draw_info[[1]]
  if (is.null(rownames(vote_m))) {
    rownames(vote_m) <- sapply(1:nrow(vote_m), function(i) {
      paste("resp", i, sep = "_")
    })
  }
  leg_names <- sapply(rownames(vote_m), function(name) {paste(name, "beta", sep = "_")})
  if (is.null(colnames(vote_m))) {
    colnames(vote_m) <- sapply(1:ncol(vote_m), function(i) {
      paste("q", i, sep = "_")
    })
  }
  alpha_vote_names_lower <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(2:K - 1, function(k) {
      paste(name, "alpha", "lower", k, sep = "_")
    })
  }))
  alpha_vote_names_upper <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(K:2 - 1, function(k) {
      paste(name, "alpha", "upper", k, sep = "_")
    })
  }))
  delta_vote_names_lower <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(2:K - 1, function(k) {
      paste(name, "delta", "lower", k, sep = "_")
    })
  }))
  delta_vote_names_upper <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(K:2 - 1, function(k) {
      paste(name, "delta", "upper", k, sep = "_")
    })
  }))
  colnames(all_param_draw) <-
    c(leg_names, alpha_vote_names_lower, alpha_vote_names_upper,
      delta_vote_names_lower, delta_vote_names_upper)

  return(c(list("param_draws" = all_param_draw), draw_info[-1]))
}

#' @importFrom Rcpp evalCpp
#' @export sample_ordinal_utility_probit_flip_rcpp
sample_ordinal_utility_probit_flip_rcpp <- function(
  vote_m, K, leg_mean, leg_s, alpha_mean, alpha_cov_s,
  delta_mean, delta_cov_s, num_iter = 2000, start_iter = 0, keep_iter = 1,
  leg_pos_init = NULL, alpha_pos_init = NULL, delta_pos_init = NULL,
  y_star_m = NULL, pos_ind = 0, neg_ind = 0, start_val = NULL) {

  total_iter = (num_iter - start_iter) %/% keep_iter
  init_info <- init_data_rcpp(
    vote_m, K, leg_pos_init, alpha_pos_init, delta_pos_init,
    y_star_m, total_iter)

  if (!is.null(start_val)) {
    init_info[[2]][1,] <- start_val
  }

  draw_info <- sample_ordinal_utility_probit_flip(
    init_info[[1]][,3], init_info[[1]][,1], init_info[[1]][,2],
    init_info[[2]], init_info[[3]], init_info[[4]],
    init_info[[5]], init_info[[6]],
    init_info[[7]], init_info[[8]], leg_mean, leg_s,
    alpha_mean, alpha_cov_s, delta_mean, delta_cov_s,
    num_iter, start_iter, keep_iter, pos_ind - 1, neg_ind - 1)

  all_param_draw = draw_info[[1]]
  if (is.null(rownames(vote_m))) {
    rownames(vote_m) <- sapply(1:nrow(vote_m), function(i) {
      paste("resp", i, sep = "_")
    })
  }
  leg_names <- sapply(rownames(vote_m), function(name) {paste(name, "beta", sep = "_")})
  if (is.null(colnames(vote_m))) {
    colnames(vote_m) <- sapply(1:ncol(vote_m), function(i) {
      paste("q", i, sep = "_")
    })
  }
  alpha_vote_names_lower <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(2:K - 1, function(k) {
      paste(name, "alpha", "lower", k, sep = "_")
    })
  }))
  alpha_vote_names_upper <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(K:2 - 1, function(k) {
      paste(name, "alpha", "upper", k, sep = "_")
    })
  }))
  delta_vote_names_lower <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(2:K - 1, function(k) {
      paste(name, "delta", "lower", k, sep = "_")
    })
  }))
  delta_vote_names_upper <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(K:2 - 1, function(k) {
      paste(name, "delta", "upper", k, sep = "_")
    })
  }))
  colnames(all_param_draw) <-
    c(leg_names, alpha_vote_names_lower, alpha_vote_names_upper,
      delta_vote_names_lower, delta_vote_names_upper)

  return(c(list("param_draws" = all_param_draw), draw_info[-1]))
}

#' @importFrom Rcpp evalCpp
#' @export sample_ordinal_utility_probit_flip_prior_rcpp
sample_ordinal_utility_probit_flip_prior_rcpp <- function(
  vote_m, K, leg_mean, leg_s, alpha_mean, alpha_cov_s,
  delta_mean, delta_cov_s, num_iter = 2000, start_iter = 0, keep_iter = 1,
  leg_pos_init = NULL, alpha_pos_init = NULL, delta_pos_init = NULL,
  y_star_m = NULL, pos_ind = 0, neg_ind = 0, start_val = NULL) {

  total_iter = (num_iter - start_iter) %/% keep_iter
  init_info <- init_data_rcpp(
    vote_m, K, leg_pos_init, alpha_pos_init, delta_pos_init,
    y_star_m, total_iter)

  if (!is.null(start_val)) {
    init_info[[2]][1,] <- start_val
  }

  draw_info <- sample_ordinal_utility_probit_flip_prior(
    init_info[[1]][,3], init_info[[1]][,1], init_info[[1]][,2],
    init_info[[2]], init_info[[3]], init_info[[4]],
    init_info[[5]], init_info[[6]],
    init_info[[7]], init_info[[8]], leg_mean, leg_s,
    alpha_mean, alpha_cov_s, delta_mean, delta_cov_s,
    num_iter, start_iter, keep_iter, pos_ind - 1, neg_ind - 1)

  all_param_draw = draw_info[[1]]
  if (is.null(rownames(vote_m))) {
    rownames(vote_m) <- sapply(1:nrow(vote_m), function(i) {
      paste("resp", i, sep = "_")
    })
  }
  leg_names <- sapply(rownames(vote_m), function(name) {paste(name, "beta", sep = "_")})
  if (is.null(colnames(vote_m))) {
    colnames(vote_m) <- sapply(1:ncol(vote_m), function(i) {
      paste("q", i, sep = "_")
    })
  }
  alpha_vote_names_lower <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(2:K - 1, function(k) {
      paste(name, "alpha", "lower", k, sep = "_")
    })
  }))
  alpha_vote_names_upper <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(K:2 - 1, function(k) {
      paste(name, "alpha", "upper", k, sep = "_")
    })
  }))
  delta_vote_names_lower <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(2:K - 1, function(k) {
      paste(name, "delta", "lower", k, sep = "_")
    })
  }))
  delta_vote_names_upper <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(K:2 - 1, function(k) {
      paste(name, "delta", "upper", k, sep = "_")
    })
  }))
  colnames(all_param_draw) <-
    c(leg_names, alpha_vote_names_lower, alpha_vote_names_upper,
      delta_vote_names_lower, delta_vote_names_upper)

  return(c(list("param_draws" = all_param_draw), draw_info[-1]))
}

#' @export sample_ordinal_utility_probit_gen_choices_rcpp
sample_ordinal_utility_probit_gen_choices_rcpp <- function(
  vote_m, K, num_choices_m, leg_mean, leg_s, alpha_mean, alpha_cov_s,
  delta_mean, delta_cov_s,
  num_iter = 2000, start_iter = 0, keep_iter = 1,
  leg_pos_init = NULL, alpha_pos_init = NULL, delta_pos_init = NULL,
  y_star_m = NULL, pos_ind = 0, neg_ind = 0, start_val = NULL) {

  total_iter = (num_iter - start_iter) %/% keep_iter
  init_info <- init_data_rcpp_gen_choices(
    vote_m, K, num_choices_m, leg_pos_init, alpha_pos_init, delta_pos_init,
    y_star_m, total_iter)

  if (!is.null(start_val)) {
    init_info[[2]][1,] <- start_val
  }

  # arma::uvec vote_v, arma::uvec respondent_v, arma::uvec question_v,
  # arma::uvec question_num_choices_m1_v,
  # arma::mat all_param_draws, arma::mat y_star_m,
  draw_info <- sample_ordinal_utility_probit_gen_choices(
    init_info[[1]][,3], init_info[[1]][,1], init_info[[1]][,2],
    num_choices_m[1,] - 1,
    init_info[[2]], init_info[[3]], init_info[[4]],
    init_info[[5]], init_info[[6]],
    init_info[[7]], init_info[[8]], leg_mean, leg_s,
    alpha_mean, alpha_cov_s, delta_mean, delta_cov_s,
    num_iter, start_iter, keep_iter, pos_ind - 1, neg_ind - 1)

  all_param_draw = draw_info[[1]]
  if (is.null(rownames(vote_m))) {
    rownames(vote_m) <- sapply(1:nrow(vote_m), function(i) {
      paste("resp", i, sep = "_")
    })
  }
  leg_names <- sapply(rownames(vote_m), function(name) {paste(name, "beta", sep = "_")})
  if (is.null(colnames(vote_m))) {
    colnames(vote_m) <- sapply(1:ncol(vote_m), function(i) {
      paste("q", i, sep = "_")
    })
  }
  alpha_vote_names_lower <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(2:K - 1, function(k) {
      paste(name, "alpha", "lower", k, sep = "_")
    })
  }))
  alpha_vote_names_upper <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(K:2 - 1, function(k) {
      paste(name, "alpha", "upper", k, sep = "_")
    })
  }))
  delta_vote_names_lower <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(2:K - 1, function(k) {
      paste(name, "delta", "lower", k, sep = "_")
    })
  }))
  delta_vote_names_upper <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(K:2 - 1, function(k) {
      paste(name, "delta", "upper", k, sep = "_")
    })
  }))
  colnames(all_param_draw) <-
    c(leg_names, alpha_vote_names_lower, alpha_vote_names_upper,
      delta_vote_names_lower, delta_vote_names_upper)

  return(c(list("param_draws" = all_param_draw), draw_info[-1]))
}

#' @export sample_ordinal_utility_probit_gen_choices_flip_alpha_rcpp
sample_ordinal_utility_probit_gen_choices_flip_alpha_rcpp <- function(
  vote_m, K, num_choices_m, leg_mean, leg_s, alpha_mean, alpha_cov_s,
  delta_mean, delta_cov_s,
  num_iter = 2000, start_iter = 0, keep_iter = 1,
  leg_pos_init = NULL, alpha_pos_init = NULL, delta_pos_init = NULL,
  y_star_m = NULL, pos_ind = 0, neg_ind = 0, start_val = NULL) {

  total_iter = (num_iter - start_iter) %/% keep_iter
  init_info <- init_data_rcpp_gen_choices(
    vote_m, K, num_choices_m, leg_pos_init, alpha_pos_init, delta_pos_init,
    y_star_m, total_iter)

  if (!is.null(start_val)) {
    init_info[[2]][1,] <- start_val
  }

  # arma::uvec vote_v, arma::uvec respondent_v, arma::uvec question_v,
  # arma::uvec question_num_choices_m1_v,
  # arma::mat all_param_draws, arma::mat y_star_m,
  draw_info <- sample_ordinal_utility_probit_gen_choices_flip_alpha(
    init_info[[1]][,3], init_info[[1]][,1], init_info[[1]][,2],
    num_choices_m[1,] - 1,
    init_info[[2]], init_info[[3]], init_info[[4]],
    init_info[[5]], init_info[[6]],
    init_info[[7]], init_info[[8]], leg_mean, leg_s,
    alpha_mean, alpha_cov_s, delta_mean, delta_cov_s,
    num_iter, start_iter, keep_iter, pos_ind - 1, neg_ind - 1)

  all_param_draw = draw_info[[1]]
  if (is.null(rownames(vote_m))) {
    rownames(vote_m) <- sapply(1:nrow(vote_m), function(i) {
      paste("resp", i, sep = "_")
    })
  }
  leg_names <- sapply(rownames(vote_m), function(name) {paste(name, "beta", sep = "_")})
  if (is.null(colnames(vote_m))) {
    colnames(vote_m) <- sapply(1:ncol(vote_m), function(i) {
      paste("q", i, sep = "_")
    })
  }
  alpha_vote_names_lower <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(2:K - 1, function(k) {
      paste(name, "alpha", "lower", k, sep = "_")
    })
  }))
  alpha_vote_names_upper <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(K:2 - 1, function(k) {
      paste(name, "alpha", "upper", k, sep = "_")
    })
  }))
  delta_vote_names_lower <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(2:K - 1, function(k) {
      paste(name, "delta", "lower", k, sep = "_")
    })
  }))
  delta_vote_names_upper <- as.vector(sapply(colnames(vote_m), function(name) {
    sapply(K:2 - 1, function(k) {
      paste(name, "delta", "upper", k, sep = "_")
    })
  }))
  colnames(all_param_draw) <-
    c(leg_names, alpha_vote_names_lower, alpha_vote_names_upper,
      delta_vote_names_lower, delta_vote_names_upper)

  return(c(list("param_draws" = all_param_draw), draw_info[-1]))
}

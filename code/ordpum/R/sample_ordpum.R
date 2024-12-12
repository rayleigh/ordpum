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

#' @importFrom parallel mclapply
calc_log_ll <- function(
  vote_v, flip_vote_v, ind_vote_v, question_vote_v, question_m1,
  alpha_lower_vals, alpha_upper_vals,
  beta_vals, delta_lower_vals, delta_upper_vals,
  flip_alpha_lower_vals, flip_alpha_upper_vals,
  flip_beta_vals, flip_delta_lower_vals, flip_delta_upper_vals,
  max_response_num, num_cores) {

  log_ll_info <-
    mclapply(1:length(vote_v), function(i) {
      ind_num = ind_vote_v[i]
      question_num = question_vote_v[i]
      half_alpha_size = question_m1[question_num]

      interested_lower_inds <-
        1:half_alpha_size + (question_num - 1) * max_response_num
      interested_upper_inds <-
        (question_num) * max_response_num - (half_alpha_size:1 - 1)

      mean_v_lower <- alpha_lower_vals[interested_lower_inds] *
        (beta_vals[ind_num] - delta_lower_vals[interested_lower_inds])
      mean_v_upper <- alpha_upper_vals[interested_upper_inds] *
        (beta_vals[ind_num] - delta_upper_vals[interested_upper_inds])
      curr_log_ll = calc_choice_k_prob(mean_v_lower, mean_v_upper, vote_v[i])

      mean_v_lower <- flip_alpha_lower_vals[interested_lower_inds] *
        (flip_beta_vals[ind_num] - flip_delta_lower_vals[interested_lower_inds])
      mean_v_upper <- flip_alpha_upper_vals[interested_upper_inds] *
        (flip_beta_vals[ind_num] - flip_delta_upper_vals[interested_upper_inds])
      flip_log_ll = calc_choice_k_prob(mean_v_lower, mean_v_upper, flip_vote_v[i])
      c(curr_log_ll, flip_log_ll)
    }, mc.cores = num_cores)
  #log_ll_info <- t(do.call(cbind, log_ll_info))
  log_ll_info <- do.call(rbind, log_ll_info)
  log_ll_info <- pmax(log_ll_info, 1e-9)
  log_ll_info <- pmin(log_ll_info, 1 - 1e-9)
  log_ll_info <- log(log_ll_info)
  return(log_ll_info)
}

#' @importFrom magrittr %>%
#' @importFrom mvtnorm rmvnorm
flip_question_parallel <- function(
  vote_v, alpha_lower_vals, alpha_upper_vals,
  beta_vals, delta_lower_vals, delta_upper_vals,
  alpha_mean, alpha_cov_s, delta_mean, delta_cov_s,
  ind_vote_v, question_vote_v,
  max_response_num, question_m1, num_cores) {

  flip_vote_v <- vote_v
  flip_alpha_lower_vals <- rep(0, length(alpha_lower_vals))
  flip_alpha_upper_vals <- rep(0, length(alpha_upper_vals))
  flip_delta_lower_vals <- rep(0, length(delta_lower_vals))
  flip_delta_upper_vals <- rep(0, length(delta_upper_vals))
  for (i in 1:length(question_m1)) {
    half_alpha_size = question_m1[i]

    interested_inds <- which(question_vote_v == i)
    flip_vote_v[interested_inds] <-
      half_alpha_size - flip_vote_v[interested_inds]

    interested_lower_inds <-
      1:half_alpha_size + (i - 1) * max_response_num
    interested_upper_inds <-
      (i) * max_response_num - (half_alpha_size:1 - 1)

    interested_prior_inds <-
      (max_response_num - half_alpha_size + 1):
      (max_response_num + half_alpha_size)
    tmp <- sample_alpha_ordinal_independent_lower(
      alpha_mean[interested_prior_inds],
      alpha_cov_s[interested_prior_inds, interested_prior_inds])

    flip_alpha_lower_vals[interested_lower_inds] <-
      tmp[1:half_alpha_size]
    flip_alpha_upper_vals[interested_upper_inds] <-
      tmp[half_alpha_size + 1:half_alpha_size]

    tmp <- rmvnorm(1, delta_mean[interested_prior_inds],
                   delta_cov_s[interested_prior_inds, interested_prior_inds])
    flip_delta_lower_vals[interested_lower_inds] <-
      tmp[1:half_alpha_size]
    flip_alpha_upper_vals[interested_upper_inds] <-
      tmp[half_alpha_size + 1:half_alpha_size]
  }

  accept_count <- rep(0, length(question_m1))

  log_ll_info <-
    calc_log_ll(vote_v, flip_vote_v, ind_vote_v, question_vote_v, question_m1,
                alpha_lower_vals, alpha_upper_vals,
                beta_vals, delta_lower_vals, delta_upper_vals,
                flip_alpha_lower_vals, flip_alpha_upper_vals, beta_vals,
                flip_delta_lower_vals, flip_delta_upper_vals,
                max_response_num, num_cores)

  log_ll_df <- as.data.frame(log_ll_info)
  colnames(log_ll_df) <- c("curr_prob", "flip_prob")
  log_ll_df$ind <- question_vote_v

  accept_prob_m <-
    log_ll_df %>% dplyr::group_by(ind) %>%
    dplyr::summarize(
      curr_ll = sum(curr_prob),
      flip_ll = sum(flip_prob)) %>%
    dplyr::arrange(ind)

  accepted_inds <-
    (1:length(question_m1))[
        which(log(runif(nrow(accept_prob_m))) <
                             accept_prob_m$flip_ll - accept_prob_m$curr_ll)]
  log_ll_v <- log_ll_info[,1]
  if (length(accepted_inds) > 0) {

    accept_count[accepted_inds] <- accept_count[accepted_inds] + 1

    interested_q_inds <-
      which(question_vote_v %in% (accepted_inds))
    vote_v[interested_q_inds] <-
      flip_vote_v[interested_q_inds]
    log_ll_v[interested_q_inds] <- log_ll_info[interested_q_inds,2]

    interested_inds <- as.vector(sapply(accepted_inds, function(i) {
      (i - 1) * max_response_num + 1:max_response_num
    }))
    alpha_lower_vals[interested_inds] <-
      flip_alpha_lower_vals[interested_inds]
    alpha_upper_vals[interested_inds] <-
      flip_alpha_upper_vals[interested_inds]
    delta_lower_vals[interested_inds] <-
      flip_delta_lower_vals[interested_inds]
    delta_upper_vals[interested_inds] <-
      flip_delta_upper_vals[interested_inds]
  }
  return(list(vote_v, alpha_lower_vals, alpha_upper_vals,
              delta_lower_vals, delta_upper_vals,
              log_ll_v, accept_count))
}

#' @importFrom magrittr %>%
#' @importFrom stats dnorm runif
flip_beta_parallel <- function(
  vote_v, alpha_lower_vals, alpha_upper_vals,
  beta_vals, delta_lower_vals, delta_upper_vals,
  beta_mean, beta_sd, beta_flip_sd,
  ind_vote_v, question_vote_v,
  max_response_num, question_m1, num_cores,
  pos_inds, neg_inds) {

  flip_beta_vals <-
    beta_flip_sd * rnorm(length(beta_vals)) - beta_vals
  accept_count <- rep(0, length(beta_vals))

  log_ll_info <-
    calc_log_ll(vote_v, vote_v, ind_vote_v, question_vote_v, question_m1,
                alpha_lower_vals, alpha_upper_vals,
                beta_vals, delta_lower_vals, delta_upper_vals,
                alpha_lower_vals, alpha_upper_vals, flip_beta_vals,
                delta_lower_vals, delta_upper_vals,
                max_response_num, num_cores)
  #log_ll_info <- t(do.call(cbind, log_ll_info))
  #log_ll_info <- pmax(log_ll_info, 1e-9)
  #log_ll_info <- pmin(log_ll_info, 1 - 1e-9)
  #log_ll_info <- log(log_ll_info)

  flip_beta_v_inds <- sort(unique(ind_vote_v))
  log_ll_df <- as.data.frame(log_ll_info)
  colnames(log_ll_df) <- c("curr_prob", "flip_prob")
  log_ll_df$ind <- ind_vote_v

  accept_prob_m <-
    log_ll_df %>% dplyr::group_by(ind) %>%
    dplyr::summarize(
      curr_ll = sum(curr_prob),
      flip_ll = sum(flip_prob)) %>%
    dplyr::arrange(ind)

  accept_prob_m$curr_ll <-
    accept_prob_m$curr_ll +
      dnorm(beta_vals[flip_beta_v_inds],
            beta_mean, beta_sd, log = T)

  accept_prob_m$flip_ll <-
    accept_prob_m$flip_ll +
    dnorm(flip_beta_vals[flip_beta_v_inds],
          beta_mean, beta_sd, log = T)

  accepted_inds <-
    flip_beta_v_inds[which(log(runif(nrow(accept_prob_m))) <
                             accept_prob_m$flip_ll - accept_prob_m$curr_ll)]

  accepted_inds <-
    accepted_inds[!(accepted_inds %in% pos_inds |
      accepted_inds %in% neg_inds)]

  log_ll_v <- log_ll_info[,1]
  if (length(accepted_inds) > 0) {
    beta_vals[accepted_inds] <- flip_beta_vals[accepted_inds]
    accept_count[accepted_inds] <- accept_count[accepted_inds] + 1
    log_ll_v[which(ind_vote_v %in% accepted_inds)] <-
      log_ll_info[which(ind_vote_v %in% accepted_inds),2]
  }
  return(list(beta_vals, log_ll_v, accept_count))
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

#' @export sample_ordinal_utility_probit_gen_choices_flip_resp_rcpp
sample_ordinal_utility_probit_gen_choices_flip_resp_rcpp <- function(
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
  draw_info <- sample_ordinal_utility_probit_gen_choices_flip_responses(
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

#' @export sample_ordinal_utility_probit_gen_choices_flip_resp_prior_rcpp
sample_ordinal_utility_probit_gen_choices_flip_resp_prior_rcpp <- function(
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
  draw_info <- sample_ordinal_utility_probit_gen_choices_flip_responses_prior(
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

#' @export sample_ordinal_utility_probit_gen_choices_flip_beta_rcpp
sample_ordinal_utility_probit_gen_choices_flip_beta_rcpp <- function(
  vote_m, K, num_choices_m, leg_mean, leg_s, alpha_mean, alpha_cov_s,
  delta_mean, delta_cov_s,
  num_iter = 2000, start_iter = 0, keep_iter = 1,
  leg_pos_init = NULL, alpha_pos_init = NULL, delta_pos_init = NULL,
  y_star_m = NULL, pos_ind = 0, neg_ind = 0, start_val = NULL,
  flip_beta_v = 1:nrow(vote_m), flip_beta_sd = 1, num_cores = 40) {

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
  #draw_info <- sample_ordinal_utility_probit_gen_choices_flip_beta(
  draw_info <- sample_ordinal_utility_probit_gen_choices_flip_beta_parallel(
    init_info[[1]][,3], init_info[[1]][,1], init_info[[1]][,2],
    num_choices_m[1,] - 1,
    init_info[[2]], init_info[[3]], init_info[[4]],
    init_info[[5]], init_info[[6]],
    init_info[[7]], init_info[[8]], leg_mean, leg_s,
    alpha_mean, alpha_cov_s, delta_mean, delta_cov_s,
    num_iter, start_iter, keep_iter, pos_ind - 1, neg_ind - 1,
    flip_beta_v - 1, flip_beta_sd, num_cores)

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

#' @export sample_ordinal_utility_probit_gen_choices_flip_beta_R_rcpp
sample_ordinal_utility_probit_gen_choices_flip_beta_R_rcpp <- function(
  vote_m, K, num_choices_m, leg_mean, leg_s, alpha_mean, alpha_cov_s,
  delta_mean, delta_cov_s,
  num_iter = 2000, start_iter = 0, keep_iter = 1,
  leg_pos_init = NULL, alpha_pos_init = NULL, delta_pos_init = NULL,
  y_star_m = NULL, pos_ind = 0, neg_ind = 0, start_val = NULL,
  flip_beta_v = 1:nrow(vote_m), flip_beta_sd = 1, num_cores = 2) {

  total_iter = (num_iter - start_iter) %/% keep_iter
  init_info <- init_data_rcpp_gen_choices(
    vote_m, K, num_choices_m, leg_pos_init, alpha_pos_init, delta_pos_init,
    y_star_m, total_iter)

  if (!is.null(start_val)) {
    init_info[[2]][1,] <- start_val
  }

  all_param_draw <- init_info[[2]]
  start_draw <- init_info[[2]][1,]
  y_star_m <- init_info[[3]]
  iter_run_num = 0
  draw_iter = 1
  between_draw_iter_num = keep_iter
  y_star_m_log_ll_m <- rep(0, ncol = total_iter)
  accept_count <- rep(0, nrow(vote_m))
  accept_count_response <- rep(0, ncol(vote_m))
  log_ll <- matrix(0, ncol = length(init_info[[1]][,3]),
                   nrow = num_iter %/% 50)
  while(iter_run_num < num_iter) {

    if (iter_run_num %% 100 == 0) {
      print(iter_run_num)
    }

    next_run_iter = min(num_iter - iter_run_num, 50)
    tmp <- matrix(start_draw, nrow = next_run_iter, ncol = length(start_draw),
                  byrow = T)

    # arma::uvec vote_v, arma::uvec respondent_v, arma::uvec question_v,
    # arma::uvec question_num_choices_m1_v,
    # arma::mat all_param_draws, arma::mat y_star_m,
    # int leg_start_ind, int alpha_v_lower_start_ind, int alpha_v_upper_start_ind,
    # int delta_v_lower_start_ind, int delta_v_upper_start_ind,
    # double leg_mean, double leg_sd, arma::vec alpha_mean_v, arma::mat alpha_cov_s,
    # arma::vec delta_mean_v, arma::mat delta_cov_s,
    # int num_iter, int start_iter, int keep_iter, int pos_ind, int neg_ind
    draw_info <- sample_ordinal_utility_probit_gen_choices(
      init_info[[1]][,3], init_info[[1]][,1], init_info[[1]][,2],
      num_choices_m[1,] - 1,
      tmp, y_star_m, init_info[[4]],
      init_info[[5]], init_info[[6]],
      init_info[[7]], init_info[[8]], leg_mean, leg_s,
      alpha_mean, alpha_cov_s, delta_mean, delta_cov_s,
      next_run_iter, 0, 1, pos_ind - 1, neg_ind - 1)

    if (next_run_iter == 50) {
      
      last_draw <- draw_info[[1]][nrow(draw_info[[1]]),]
      flip_info <- flip_question_parallel(
        init_info[[1]][,3],
        last_draw[init_info[[5]] + 1:(ncol(vote_m) * (K - 1))],
        last_draw[init_info[[6]] + 1:(ncol(vote_m) * (K - 1))],
        last_draw[init_info[[4]] + 1:(nrow(vote_m))],
        last_draw[init_info[[7]] + 1:(ncol(vote_m) * (K - 1))],
        last_draw[init_info[[8]] + 1:(ncol(vote_m) * (K - 1))],
        alpha_mean, alpha_cov_s, delta_mean, delta_cov_s,
        init_info[[1]][,1] + 1, init_info[[1]][,2] + 1,
        K - 1, num_choices_m[1,] - 1, num_cores)

      init_info[[1]][,3] <- flip_info[[1]]
      draw_info[[1]][next_run_iter, init_info[[5]] + 1:(ncol(vote_m) * (K - 1))] <-
        flip_info[[2]]
      draw_info[[1]][next_run_iter, init_info[[6]] + 1:(ncol(vote_m) * (K - 1))] <-
        flip_info[[3]]
      draw_info[[1]][next_run_iter, init_info[[7]] + 1:(ncol(vote_m) * (K - 1))] <-
        flip_info[[4]]
      draw_info[[1]][next_run_iter, init_info[[8]] + 1:(ncol(vote_m) * (K - 1))] <-
        flip_info[[5]]

      log_ll[iter_run_num %/% 50 + 1,] <- flip_info[[length(flip_info) - 1]]
      accept_count_response = accept_count_response +
        flip_info[[length(flip_info)]]

      interested_inds <- which(init_info[[1]][,1] %in% (flip_beta_v - 1))
      last_draw <- draw_info[[1]][nrow(draw_info[[1]]),]
      flip_info <- flip_beta_parallel(
        init_info[[1]][interested_inds,3],
        last_draw[init_info[[5]] + 1:(ncol(vote_m) * (K - 1))],
        last_draw[init_info[[6]] + 1:(ncol(vote_m) * (K - 1))],
        last_draw[init_info[[4]] + 1:(nrow(vote_m))],
        last_draw[init_info[[7]] + 1:(ncol(vote_m) * (K - 1))],
        last_draw[init_info[[8]] + 1:(ncol(vote_m) * (K - 1))],
        leg_mean, leg_s, flip_beta_sd,
        init_info[[1]][interested_inds,1] + 1, 
	init_info[[1]][interested_inds,2] + 1,
        K - 1, num_choices_m[1,] - 1, num_cores,
        pos_ind, neg_ind)

      draw_info[[1]][next_run_iter,1:nrow(vote_m)] <- flip_info[[1]]
      log_ll[iter_run_num %/% 50 + 1, interested_inds] <- flip_info[[2]]
      accept_count = accept_count + flip_info[[3]]

    }

    iter_run_num = iter_run_num + next_run_iter
    start_draw = draw_info[[1]][next_run_iter,]
    y_star_m = draw_info[[2]]

    if (iter_run_num > start_iter) {
      iter_diff = min(iter_run_num - start_iter, 50)
      while (between_draw_iter_num <= iter_diff) {

        all_param_draw[draw_iter,] = draw_info[[1]][between_draw_iter_num,]
        y_star_m_log_ll_m[draw_iter] = draw_info[[3]][between_draw_iter_num]

        between_draw_iter_num = between_draw_iter_num + keep_iter
        draw_iter = draw_iter + 1
      }
      between_draw_iter_num = between_draw_iter_num - iter_diff
    }

  }

  # all_param_draw = draw_info[[1]]
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

  return(c(list("param_draws" = all_param_draw,
                "y_star_m" = y_star_m,
                "y_star_m_log_ll" = y_star_m_log_ll_m,
                "accept_count_v" = accept_count,
		"accept_count_response_v" = accept_count_response,
                "log_ll" = log_ll)))
}

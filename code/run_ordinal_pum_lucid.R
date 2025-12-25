library(Rcpp)
library(msm)
library(mvtnorm)
library(ordpum)
library(LaplacesDemon)

load("data_files/lucid/lucid_responses_fixed_order.Rdata")

beta_init <- rnorm(nrow(responses))
starting_signs <- 1

alpha_init_lower <- matrix(rtnorm(ncol(responses) * 4, lower = rep(0, 4)), nrow = 4)
alpha_init_lower <- as.vector(
  apply(sweep(alpha_init_lower, 2, -starting_signs, "*"), 2, sort))

alpha_init_upper <- matrix(rtnorm(ncol(responses) * 4, lower = rep(0, 4)), nrow = 4)
alpha_init_upper <- as.vector(
  apply(sweep(alpha_init_upper, 2, starting_signs, "*"), 2, sort))

delta_init_lower <- rmvnorm(ncol(responses), rep(0, 4), diag(4))
delta_init_lower <- as.vector(t(delta_init_lower * -starting_signs))

delta_init_upper <- rmvnorm(ncol(responses), rep(0, 4), diag(4))
delta_init_upper <- as.vector(t(delta_init_upper * starting_signs))

delta_mean_v_lower <- -4 + 4:1 * -1.5
delta_mean_v_upper <- 1:4 * 1.5 + 4
delta_sigma_cov = 25 * diag(8)
alpha_sigma_cov = 100 * diag(8)

chain_run <- sample_ordinal_utility_probit_gen_choices_flip_beta_R_rcpp(
  responses + 1, 5,
  matrix(5, nrow = nrow(responses), ncol = ncol(responses)),
  0, 1, rep(0, 8), alpha_sigma_cov,
  c(delta_mean_v_lower, delta_mean_v_upper), delta_sigma_cov,
  num_iter = 110000, start_iter = 10000, keep_iter = 10,
  pos_ind = 2562,
  leg_pos_init = beta_init,
  alpha_pos_init = rbind(alpha_init_lower, alpha_init_upper),
  delta_pos_init = rbind(delta_init_lower, delta_init_upper),
  flip_beta_sd = beta_sd, num_cores = 80)

y_star_m <- chain_run[[2]]
start_v <- chain_run[[1]][nrow(chain_run[[1]]),]
bimodal_check <- 
  which(apply(chain_run[[1]][, grep("beta", colnames(chain_run[[1]]))], 2, function(col) is.multimodal(col, 0.01)))
#print(length(bimodal_check))
rm(chain_run)

#load("multi_betas_ind_to_flip_corrected_order.Rdata")
#bimodal_check <- sort(unique(unlist(multi_betas_list)))

chain_run <- sample_ordinal_utility_probit_gen_choices_flip_beta_R_rcpp(
  responses + 1, 5,
  matrix(5, nrow = nrow(responses), ncol = ncol(responses)),
  0, 1, rep(0, 8), alpha_sigma_cov,
  c(delta_mean_v_lower, delta_mean_v_upper), delta_sigma_cov,
  num_iter = 300000, start_iter = 200000, keep_iter = 10,
  pos_ind = 2562,
  leg_pos_init = beta_init,
  alpha_pos_init = rbind(alpha_init_lower, alpha_init_upper),
  delta_pos_init = rbind(delta_init_lower, delta_init_upper),
  flip_beta_v = bimodal_check, flip_beta_sd = beta_sd, num_cores = 80,
  y_star_m = y_star_m, start_val = start_v)

save(chain_run, file = paste0("result_files/ordinal_pum_lucid_results.Rdata"))

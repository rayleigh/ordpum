library(cmdstanr)

load("data_files/lucid/lucid_responses_fixed_order.Rdata")

mod <- cmdstan_model("bayesian_graded_response_model.stan")
data_m <- expand.grid(1:nrow(responses), 1:ncol(responses))
data_m$responses <- as.vector(responses) + 1
data_m <- data_m[!is.na(data_m$responses),]
data_list = list("K" = 5,
                 "num_ind" = nrow(responses),
                 "num_responses" = ncol(responses),
                 "total_responses" = nrow(data_m),
                 "response_v" = data_m[,3],
                 "response_ind_v" = data_m[,1],
                 "response_q_v" = data_m[,2],
                 "alpha_sd" = 0.5,
                 "delta_mu" = c(-4, -2, 2, 4))

grm_stan_fit <- mod$sample(data = data_list, parallel_chains = 4, 
                           iter_warmup = 10000, iter_sampling = 100000, thin = 10)
save(grm_stan_fit, file = "ordinal_utility/result_files/bgrm/bgrm_stan_output_correct_order_2.Rdata")


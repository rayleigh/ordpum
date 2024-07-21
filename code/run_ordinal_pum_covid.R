library(ordpum)

responses <- read.csv("data_files/covid_survey/COVID_survey_refined.csv")
interested_questions <- c(sapply(c(2, 3, 21, 24, 26), function(i) {
  paste0("Q", i, "_")
}), c("Q12", "Q13", "Q23"))
interested_cols <- unlist(sapply(interested_questions, function(q_name) {
  grep(q_name, colnames(responses))
}))
responses <- data.matrix(responses[, interested_cols])

responses[, 30] <- 7 - responses[, 30]
responses[, c(22, 27)] <- 5 - responses[, c(22, 27)]

remove_ind <-
  which(apply(responses, 1, function(row) length(unique(row[!is.na(row)]))) < 3)
remove_ind <-
  c(remove_ind,
    which(apply(responses, 1, function(row) mean(is.na(row))) > 0.4))
responses <- responses[-remove_ind,]

max_question_num <-
  matrix(4, nrow = nrow(responses), ncol = ncol(responses))
colnames(max_question_num) <- responses
max_question_num <- 4
max_question_num[, grep("Q23", colnames(responses))] = 6
max_question_num[, grep("Q12", colnames(responses))] = 3
max_question_num[, grep("Q24_", colnames(responses))] = 3
max_question_num[, grep("Q3_", colnames(responses))] = 3


#set.seed(123 + rand_inits)
beta_init <- rnorm(nrow(responses))

max_q_num = 5

alpha_init_lower <- matrix(rtnorm(ncol(responses) * max_q_num, lower = rep(0, max_q_num)), nrow = max_q_num)
for (i in 1:ncol(max_question_num)) {
  alpha_init_lower[,i] <- sort(-alpha_init_lower[,i])
  if (max_question_num[1, i] != 6) {
    alpha_init_lower[max_question_num[1, i]:5,i] = 0
  }
}
alpha_init_lower <- as.vector(alpha_init_lower)

alpha_init_upper <- matrix(rtnorm(ncol(responses) * max_q_num, lower = rep(0, max_q_num)), nrow = max_q_num)
for (i in 1:ncol(max_question_num)) {
  alpha_init_upper[,i] <- sort(alpha_init_upper[,i])
  if (max_question_num[1, i] != 6) {
    alpha_init_upper[1:(6 - max_question_num[1, i]),i] = 0
  }
}
alpha_init_upper <- as.vector(alpha_init_upper)

delta_init_lower <- rmvnorm(ncol(responses), rep(0, 5), diag(5))
for (i in 1:ncol(max_question_num)) {
  if (max_question_num[1, i] != 6) {
    delta_init_lower[i, max_question_num[1, i]:5] = 0
  }
}
delta_init_lower <- as.vector(t(delta_init_lower * -1))

delta_init_upper <- rmvnorm(ncol(responses), rep(0, 5), diag(5))
for (i in 1:ncol(max_question_num)) {
  if (max_question_num[1, i] != 6) {
    delta_init_upper[i, 1:(6 - max_question_num[1, i])] = 0
  }
}
delta_init_upper <- as.vector(t(delta_init_upper * 1))

delta_mean_v_lower <- -4 + 5:1 * -1.5
delta_mean_v_upper <- 1:5 * 1.5 + 4
delta_sigma_cov = 25 * diag(10)
alpha_sigma_cov = 100 * diag(10)

chain_run <- sample_ordinal_utility_probit_gen_choices_rcpp(
  responses, 6, max_question_num,
  0, 1, rep(0, 10), alpha_sigma_cov,
  c(delta_mean_v_lower, delta_mean_v_upper), delta_sigma_cov,
  leg_pos_init = beta_init,
  alpha_pos_init = rbind(alpha_init_lower, alpha_init_upper),
  delta_pos_init = rbind(delta_init_lower, delta_init_upper),
  num_iter = 200000, start_iter = 100000, keep_iter = 10)
save(chain_run, file = paste0("result_files/ordinal_pum_covid_results.Rdata"))

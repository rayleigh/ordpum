library(Rcpp)
library(msm)
library(mvtnorm)
library(ordpum)

responses <- read.csv("../data_files/lucid/lucid_data.csv")
## It will be useful to record the question wording for later
questions <- responses[1, ]
QUESTIONS <- gsub("\n", " ", questions[ , grepl("^IMM", colnames(questions))])
## Eliminate respondents who do not pass the attention checks
feelings  <- which(names(responses) == "SCREENER_FEELINGS")
correct1  <- responses[ , feelings] == "Proud,None of the above"
interest  <- which(names(responses) == "SCREENER_INTEREST")
correct2  <- responses[ , interest] == "Extremely interested,Not interested at all"
colors    <- which(names(responses) == "SCREENER_COLORS")
correct3  <- responses[ , colors] == "Red,Green"
attentive <- correct1 & correct2 & correct3
responses <- responses[attentive, ]
## Get the ideology and party ID variables to validate
vinfo     <- responses[ , c(27:30, 32)]
party_id  <- (vinfo[ , "PID"] == "Republican") - (vinfo[ , "PID"] == "Democrat")
party_id  <- party_id + (vinfo[ , "R_STRENGTH"] == "A strong Republican")
party_id  <- party_id - (vinfo[ , "D_STRENGTH"] == "A strong Democrat")
party_id  <- party_id + (vinfo[ , "LEANERS"] == "Republican")
party_id  <- party_id - (vinfo[ , "LEANERS"] == "Democratic")
ideo_levs <- c("Very liberal", "Somewhat liberal", "Slightly liberal",
               "Moderate; middle of the road", "Slightly conservative",
               "Somewhat conservative", "Very conservative")
ideo      <- as.integer(factor(vinfo[ , "IDEO"], levels = ideo_levs))
vinfo     <- data.frame(party_id = party_id, ideo = ideo)
## Get just the responses to the immigration battery
responses <- responses[ , grepl("^IMM", colnames(responses))]
## Code the responses in {NA, 0, 1, 2, 3, 4}
q_options <- c(paste(c("Strongly", "Somewhat"), "disagree"),
               "Neither disagree nor agree",
               paste(c("Strongly", "Somewhat"), "agree"))
responses <- apply(responses, 2, match, q_options) - 1
## Eliminate respondents who straight-line
str8lines <- apply(responses, 1, function(x) all(x %in% 0:2) | all(x %in% 2:4))
responses <- responses[!str8lines, ]
vinfo     <- vinfo[!str8lines, ]

#set.seed(123 + rand_inits)
beta_init <- rnorm(nrow(responses))
starting_signs <- 2 * rbinom(10, 1, 1/2) - 1

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

chain_run <- sample_ordinal_utility_probit_flip_prior_rcpp(
  responses + 1, 5, 0, 1, rep(0, 8), alpha_sigma_cov,
  c(delta_mean_v_lower, delta_mean_v_upper), delta_sigma_cov,
  num_iter = 50000, start_iter = 0, keep_iter = 10,
  pos_ind = 2562, leg_pos_init = beta_init, 
  alpha_pos_init = rbind(alpha_init_lower, alpha_init_upper), 
  delta_pos_init = rbind(delta_init_lower, delta_init_upper))
save(chain_run, file = paste0("result_files/ordinal_pum_lucid_results.Rdata"))

# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

sample_alpha_ordinal_independent_lower <- function(alpha_post_mean_m, alpha_post_cov_s, num_iter = 20L) {
    .Call(`_ordpum_sample_alpha_ordinal_independent_lower`, alpha_post_mean_m, alpha_post_cov_s, num_iter)
}

calc_choice_k_prob <- function(mean_1, mean_2, choice_k) {
    .Call(`_ordpum_calc_choice_k_prob`, mean_1, mean_2, choice_k)
}

calc_log_ll_y_star <- function(vote_v, y_star_m, alpha_v_lower, alpha_v_upper, beta_v, delta_v_lower, delta_v_upper, respondent_v, question_v, question_num_choices_m1_v) {
    .Call(`_ordpum_calc_log_ll_y_star`, vote_v, y_star_m, alpha_v_lower, alpha_v_upper, beta_v, delta_v_lower, delta_v_upper, respondent_v, question_v, question_num_choices_m1_v)
}

sample_ordinal_utility_probit <- function(vote_v, respondent_v, question_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind) {
    .Call(`_ordpum_sample_ordinal_utility_probit`, vote_v, respondent_v, question_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind)
}

sample_ordinal_utility_probit_gen_choices <- function(vote_v, respondent_v, question_v, question_num_choices_m1_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind) {
    .Call(`_ordpum_sample_ordinal_utility_probit_gen_choices`, vote_v, respondent_v, question_v, question_num_choices_m1_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind)
}

sample_ordinal_utility_probit_flip <- function(vote_v, respondent_v, question_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind) {
    .Call(`_ordpum_sample_ordinal_utility_probit_flip`, vote_v, respondent_v, question_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind)
}

sample_ordinal_utility_probit_flip_prior <- function(vote_v, respondent_v, question_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind) {
    .Call(`_ordpum_sample_ordinal_utility_probit_flip_prior`, vote_v, respondent_v, question_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind)
}

sample_ordinal_utility_probit_gen_choices_flip_responses <- function(vote_v, respondent_v, question_v, question_num_choices_m1_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind) {
    .Call(`_ordpum_sample_ordinal_utility_probit_gen_choices_flip_responses`, vote_v, respondent_v, question_v, question_num_choices_m1_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind)
}

sample_ordinal_utility_probit_gen_choices_flip_responses_prior <- function(vote_v, respondent_v, question_v, question_num_choices_m1_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind) {
    .Call(`_ordpum_sample_ordinal_utility_probit_gen_choices_flip_responses_prior`, vote_v, respondent_v, question_v, question_num_choices_m1_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind)
}

sample_ordinal_utility_probit_gen_choices_flip_beta <- function(vote_v, respondent_v, question_v, question_num_choices_m1_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind, flip_beta_v, flip_beta_sd) {
    .Call(`_ordpum_sample_ordinal_utility_probit_gen_choices_flip_beta`, vote_v, respondent_v, question_v, question_num_choices_m1_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind, flip_beta_v, flip_beta_sd)
}

sample_ordinal_utility_probit_gen_choices_flip_beta_parallel <- function(vote_v, respondent_v, question_v, question_num_choices_m1_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind, flip_beta_v, flip_beta_sd, num_cores) {
    .Call(`_ordpum_sample_ordinal_utility_probit_gen_choices_flip_beta_parallel`, vote_v, respondent_v, question_v, question_num_choices_m1_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind, flip_beta_v, flip_beta_sd, num_cores)
}

sample_ordinal_utility_probit_gen_choices_flip_alpha <- function(vote_v, respondent_v, question_v, question_num_choices_m1_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind) {
    .Call(`_ordpum_sample_ordinal_utility_probit_gen_choices_flip_alpha`, vote_v, respondent_v, question_v, question_num_choices_m1_v, all_param_draws, y_star_m, leg_start_ind, alpha_v_lower_start_ind, alpha_v_upper_start_ind, delta_v_lower_start_ind, delta_v_upper_start_ind, leg_mean, leg_sd, alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s, num_iter, start_iter, keep_iter, pos_ind, neg_ind)
}

calc_waic_ordinal_pum_utility <- function(leg_ideology, alpha_m_lower, alpha_m_upper, delta_m_lower, delta_m_upper, case_vote_m, km1) {
    .Call(`_ordpum_calc_waic_ordinal_pum_utility`, leg_ideology, alpha_m_lower, alpha_m_upper, delta_m_lower, delta_m_upper, case_vote_m, km1)
}

calc_waic_ordinal_pum_waic <- function(leg_ideology, alpha_m_lower, alpha_m_upper, delta_m_lower, delta_m_upper, case_vote_m, num_votes, km1) {
    .Call(`_ordpum_calc_waic_ordinal_pum_waic`, leg_ideology, alpha_m_lower, alpha_m_upper, delta_m_lower, delta_m_upper, case_vote_m, num_votes, km1)
}

calc_waic_ordinal_pum_block <- function(leg_ideology, alpha_m_lower, alpha_m_upper, delta_m_lower, delta_m_upper, case_vote_m, case_year, block_m) {
    .Call(`_ordpum_calc_waic_ordinal_pum_block`, leg_ideology, alpha_m_lower, alpha_m_upper, delta_m_lower, delta_m_upper, case_vote_m, case_year, block_m)
}


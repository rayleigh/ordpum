// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> K;
  int<lower=0> num_ind;
  int<lower=0> num_responses;
  int<lower=0> total_responses;
  array[total_responses] int response_v;
  array[total_responses] int response_ind_v;
  array[total_responses] int response_q_v;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  vector[num_ind] beta;
  vector<lower = 0>[num_responses] alpha;
  array[num_responses] ordered[K - 1] delta;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  beta ~ normal(0, 1);
  alpha ~ normal(0, 3);
  for (j in 1:num_responses) {
    delta[j] ~ normal(0, 3);
  }
  for (i in 1:total_responses) {
    real cat_prob_raw;
    real cat_prob_raw_lower;

   if (response_v[i] < K) {
      cat_prob_raw = (delta[response_q_v[i]][response_v[i]] -
        alpha[response_q_v[i]] * beta[response_ind_v[i]]);
    }
    if (response_v[i] > 1) {
      cat_prob_raw_lower = (delta[response_q_v[i]][response_v[i] - 1] -
        alpha[response_q_v[i]] * beta[response_ind_v[i]]);
    }
    if (response_v[i] == 1) {
      target += cat_prob_raw - log1p_exp(cat_prob_raw);
    } else if (response_v[i] == K) {
      target += -log1p_exp(cat_prob_raw_lower);
    } else {
      target += log(exp(cat_prob_raw) - exp(cat_prob_raw_lower)) -
        log1p_exp(cat_prob_raw) - log1p_exp(cat_prob_raw_lower);
    }
  }
}

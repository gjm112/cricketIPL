data {
  int<lower=1> N;               // number of observations
  int<lower=1> P;               // number of fixed effect covariates
  int<lower=1> B;               // number of batters
  int<lower=1> L;               // number of leagues
  int<lower=1> T;               // number of time points (seasons)
  int<lower=2> K;               // number of categories

  matrix[N, P] X;               // fixed effects matrix
  int<lower=1, upper=B> batter_id[N];
  int<lower=1, upper=L> league_id[N];
  int<lower=1, upper=T> season_id[N];
  int<lower=1, upper=K> y[N];   // outcome (1-based category labels)
}

parameters {
  matrix[P, K-1] beta;                   // fixed effects
  matrix[L, K-1] raw_rho;                // unconstrained AR(1) coefficients
  vector<lower=0>[L] sigma_k[K-1];       // standard deviations for each class
  cholesky_factor_corr[L] L_corr_k[K-1]; // Cholesky factors of correlation matrices
  matrix[L, K-1] eps[B, T];              // uncorrelated errors
}

transformed parameters {
  matrix[L, K-1] rho = tanh(raw_rho);  // AR(1) coefficients (constrained to (-1,1))
  matrix[L, K-1] u[B, T];            // Random Effects


  // Build covariance Cholesky factor for each category k:
  matrix[L, L] L_k[K-1];
  for (k in 1:(K-1)) {
    L_k[k] = diag_pre_multiply(sigma_k[k], L_corr_k[k]);
  }
  
  // Initial time step
  for (b in 1:B) {
    for (k in 1:(K-1)) {
      vector[L] z = eps[b, 1][, k];                 // column k
      vector[L] scaled = (1 ./ sqrt(1 - square(rho[, k]))) .* (L_k[k] * z);
      u[b, 1][, k] = scaled;
    }
  }

  // Subsequent time steps
  for (t in 2:T){
    for (b in 1:B) {
      for (k in 1:(K-1)) {
        vector[L] z = eps[b, t][, k];
        vector[L] noise = L_k[k] * z;
        u[b, t][, k] = rho[, k] .* u[b, t - 1][, k] + noise;
      }
    }
  }
}

model {
  // Priors
  to_vector(beta) ~ normal(0, 10);
  to_vector(raw_rho) ~ normal(0, 10);
  for (k in 1:(K - 1)){
    sigma_k[k] ~ normal(0, 5); 
    L_corr_k[k] ~ lkj_corr_cholesky(1);
  }
  for (b in 1:B) {
    for (t in 1:T) {
      to_vector(eps[b, t]) ~ normal(0, 1);
    }
  }

  // Likelihood
  for (n in 1:N) {
    vector[K] logits;
    logits[1] = 0;
    logits[2:K] = (X[n] * beta)' + to_vector(u[batter_id[n], season_id[n]][league_id[n]]);
    y[n] ~ categorical_logit(logits);
  }
}



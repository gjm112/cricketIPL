data {  
  int<lower=1> N; // number of observations
  int<lower=1> P; // number of fixed effect covariates
  int<lower=1> B; // number of batters
  int<lower=1> L; // number of leagues
  int<lower=2> K; // number of categories

  matrix[N, P] X;             // fixed effects matrix
  int<lower=1, upper=K> y[N]; // outcome (1-based category labels)
  
  int<lower=1> N_u;                       // number of observed (b,l,t)
  int<lower=0, upper=N_u/L - 1> u_id[N];  // maps obs n to appropriate u
  int<lower=1, upper=L>         lg_id[N]; // maps obs n to appropriate league
  
  int<lower=1> T_b[B];  // number of observed t within b
}
parameters {
  matrix[P, K-1] beta;                        // fixed effects
  array[K-1] vector<lower=-1,upper=1>[L] rho; // AR(1) coefficients
  array[K-1] vector<lower=0>[L] tau;          // standard deviations for each class
  cholesky_factor_corr[L] L_corr_chol[K-1];   // Cholesky factors of correlation matrices
  matrix[N_u, K-1] z;                       // innovations (uncorrelated)
}
transformed parameters {
  // Stationary and innovation covariance matrices
  matrix[L,L] Q[K-1];
  matrix[L,L] Sigma[K-1];
  for (k in 1:(K-1)) {
    Q[k] = diag_pre_multiply(tau[k], L_corr_chol[k]) * diag_pre_multiply(tau[k], L_corr_chol[k])';
    Sigma[k] = Q[k] ./ (rep_matrix(1.0,L,L) - rho[k] * rho[k]');
  }

  // Only store observed random effects
  matrix[N_u,K-1] u;

  // Loop through observed (b,t)
  {int i = 1; 
    for (b in 1:B) { 
      for (t in 1:T_b[b]) { 
        for (k in 1:(K-1)) { 
          if (t == 1) { 
            // First timepoint: no AR(1) history 
            u[(L*(i-1)+1):(L*i),k] = cholesky_decompose(Sigma[k]) * z[(L*(i-1)+1):(L*i),k]; 
          } else { 
            // Later timepoints: use AR(1) propagation 
            u[(L*(i-1)+1):(L*i),k] = rho[k] .* u[(L*(i-2)+1):(L*(i-1)),k] + cholesky_decompose(Q[k]) * z[(L*(i-1)+1):(L*i),k]; 
          } 
        } 
        i += 1; 
      } 
    }
  }
}
model {
  // Priors
  for (k in 1:(K-1)){
    for (p in 1:P){
      target += normal_lpdf(beta[p,k] | 0,5);
    }
    target += lkj_corr_cholesky_lpdf(L_corr_chol[k] | 2);
    for (l in 1:L){
      target += normal_lpdf(rho[k][l] | 1,1);
      target += normal_lpdf(tau[k][l] | 0,1);
    }
    for (m in 1:N_u){
      target += normal_lpdf(z[m,k] | 0,1);
    }
  }
  // Likelihood
  for (n in 1:N) {
    vector[K] logits;
    logits[1] = 0;
    for (k in 2:K) {
      // u_id[n] maps observation n to the correct (b,t) index
      logits[k] = X[n] * beta[,k-1] + u[L*u_id[n] + lg_id[n],k-1];  
    }
    y[n] ~ categorical_logit(logits);
  }
}



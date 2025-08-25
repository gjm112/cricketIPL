data {  
  int<lower=1> N; // number of observations
  int<lower=1> P; // number of fixed effect covariates
  int<lower=1> B; // number of batters in shard s
  int<lower=1> L; // number of leagues
  int<lower=2> K; // number of categories

  matrix[N, P] X;             // fixed effects matrix
  int<lower=1, upper=K> y[N]; // outcome (1-based category labels)
  
  int<lower=1> N_u;                   // number of observed (b,l,t)
  int<lower=1, upper=N_u> u_index[N]; // maps obs n to appropriate u
  
  int<lower=1> T_b[B];    // number of observed t within b
  int<lower=1> N_bt;      // number of observed (b,t)
  int<lower=1> t_b[N_bt]; // season within batter index
  
  array[N_bt] int<lower=1> n_l_bt; // total number of leagues batter b played in year t
  array[N_bt, L] int<lower=0, upper=L> idx_l_bt;  // league indices, 0-padded
  array[N_bt, L] int<lower=0, upper=N_u> idx_z; // indices for innovations z
}
parameters {
  matrix[P, K-1] beta;                        // fixed effects
  array[K-1] vector<lower=-1,upper=1>[L] rho; // AR(1) coefficients
  array[K-1] vector<lower=0>[L] tau;          // standard deviations for each class
  cholesky_factor_corr[L] L_corr_chol[K-1];   // Cholesky factors of correlation matrices
  matrix[N_u, K-1] z;                         // innovations (uncorrelated)
}
transformed parameters {
  // Compute innovation covariance matrices Q[k]
  // and stationary covariance matrix Sigma[k]
  matrix[L, L] Q[K-1];
  matrix[L, L] Sigma[K-1];
  for (k in 1:(K-1)) {
    Q[k] = diag_pre_multiply(tau[k], L_corr_chol[k]) * diag_pre_multiply(tau[k], L_corr_chol[k])';
    Sigma[k] = Q[k] ./ (rep_matrix(1.0,L,L) - rho[k] * rho[k]');
  }

  matrix[N_u, K-1] u; // Output random effects
  {int i = 1;
    
    for (b in 1:B) {
      for (t in 1:T_b[b]) {
        int n_i = n_l_bt[i];
        array[n_i] int idx_i = idx_l_bt[i,1:n_i];
        array[n_i] int idx_u = idx_z[i,1:n_i];
        matrix[n_i, n_i] Sigma_i_k;
        
        for (k in 1:(K-1)) {
          if (t == 1) {
            // First timepoint: no AR(1) history
            Sigma_i_k = Sigma[k][idx_i,idx_i];
            u[idx_u,k] = cholesky_decompose(Sigma_i_k) * z[idx_u,k];
          } else {
            // Later timepoints: use AR(1) propagation
            int t_diff = t_b[i] - t_b[i-1];
            int n_im1 = n_l_bt[i-1];
            array[n_im1] int idx_im1 = idx_l_bt[i-1,1:n_im1];
            array[n_im1] int idx_um1 = idx_z[i-1,1:n_im1];
            vector[n_i] mu_i_k = pow(rho[k][idx_i],t_diff) .* (Sigma[k][idx_i,idx_im1] * inverse(Sigma[k][idx_im1,idx_im1]) * u[idx_um1,k]);
            Sigma_i_k = quad_form_diag(Sigma[k][idx_i,idx_i] - Sigma[k][idx_i,idx_im1] * inverse(Sigma[k][idx_im1,idx_im1]) * Sigma[k][idx_im1,idx_i], pow(rho[k][idx_i],t_diff));
            for (j in 1:t_diff){
              Sigma_i_k += quad_form_diag(Q[k][idx_i,idx_i],pow(rho[k][idx_i],j-1));
            }
            u[idx_u,k] = mu_i_k + cholesky_decompose(Sigma_i_k) * z[idx_u,k];
          }
        }
        i += 1;
      }
    }
  }
}
model {
  // Priors with 1/S scaling for CMC and Hierarchical Structure
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
    logits[2:K] = (X[n] * beta)' + (u[u_index[n]])';
    y[n] ~ categorical_logit(logits);
  }
}



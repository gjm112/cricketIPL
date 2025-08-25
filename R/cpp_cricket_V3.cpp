#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>

//[[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

arma::uvec systematic_resample(const arma::vec& weights, int N) {
  // weights must sum to 1
  arma::vec cumw = arma::cumsum(weights);
  arma::uvec out(N);
  
  static thread_local std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<double> dist(0.0, 1.0/N);
  double u0 = dist(rng);
  
  int i = 0;
  for (int j = 0; j < N; ++j) {
    double uj = u0 + j*1.0/N;
    while (uj > cumw[i]) {
      i++;
    }
    out[j] = i;
  }
  
  return out;
}

//[[Rcpp::export]]
List dual_averaging(double accept_prob,
                    double target_accept,
                    int t,
                    double x,
                    double x_bar,
                    double H,
                    double mu = -2.302585,
                    double gamma = 0.05,
                    double t0 = 10.0,
                    double kappa = 0.75) {
  double g = target_accept - accept_prob;
  H = (1.0 - 1.0/(t + t0)) * H + (1.0/(t + t0)) * g;
  double x_new = mu - (std::sqrt((double)t) / gamma) * H;
  double eta = std::pow((double)t, -kappa);
  x_bar = (1.0 - eta) * x_bar + eta * x_new;
  
  return List::create(
    Named("x") = x_new,
    Named("xbar") = x_bar,
    Named("H") = H
  );
}

// [[Rcpp::export]]
double log_dnorm_vec(const arma::vec& x,
                     const arma::vec& mu,
                     double sigma) {
  int N = x.n_elem;
  
  // residuals
  arma::vec z = (x - mu) / sigma;
  
  // sum of squared standardized residuals
  double quad = dot(z, z);
  
  // sum of logs of sigma
  double log_det = -N * log(sigma);
  
  // assemble log-likelihood
  double out = -0.5 * N * log(2.0 * M_PI) - log_det - 0.5 * quad;
  
  return out;
}

// [[Rcpp::export]]
double log_dmvnorm(const arma::vec& x, 
                   const arma::vec& mu, 
                   const arma::vec& sigma,
                   const arma::mat& R) {
  int d = x.n_elem; // Dimensionality
  arma::vec diff = x - mu;
  arma::mat D = arma::diagmat(sigma);
  arma::mat Sigma = D * R * D;
  arma::mat invSigma = arma::inv_sympd(Sigma); // Inverse of covariance matrix
  double detSigma = arma::det(Sigma); // Determinant of covariance matrix
  double exponent = -0.5 * arma::as_scalar(diff.t() * invSigma * diff); // Quadratic form
  double log_norm_const = -0.5 * d * std::log(2.0 * M_PI); // Log-normalizing constant
  
  return log_norm_const - 0.5 * d * detSigma + exponent; // Return log-density if requested
}

// [[Rcpp::export]]
arma::vec dlog_dmvnorm_dsigma(const arma::vec& x, 
                               const arma::vec& mu, 
                               const arma::vec& sigma,
                               const arma::mat& R) {
  int d = sigma.n_elem;
  arma::mat D = arma::diagmat(sigma);
  arma::mat Sigma = D * R * D;
  arma::mat Sigma_inv = arma::inv_sympd(Sigma);
  
  arma::vec diff = x - mu;
  arma::mat outer = diff * diff.t();
  
  // Gradient wrt Sigma
  arma::mat G = 0.5 * Sigma_inv * (outer - Sigma) * Sigma_inv;
  arma::vec grad_sds(d, arma::fill::zeros);
  
  arma::mat DR = D * R;
  for (int i = 0; i < d; ++i) {
    // gradient wrt sigma_i
    grad_sds(i) = 2.0 * arma::dot(G.row(i), DR.row(i).t());
  }
  
  return grad_sds;
}

// [[Rcpp::export]]
arma::mat dlog_dmvnorm_dR(const arma::vec& x, 
                           const arma::vec& mu, 
                           const arma::vec& sigma,
                           const arma::mat& R) {
  arma::mat D = arma::diagmat(sigma);
  arma::mat Sigma = D * R * D;
  arma::mat Sigma_inv = arma::inv_sympd(Sigma);
  
  arma::vec diff = x - mu;
  arma::mat outer = diff * diff.t();
  arma::mat grad_R = Sigma_inv * (outer - Sigma) * Sigma_inv;
  
  grad_R *= 0.5;
  grad_R = D * grad_R * D;  // chain rule Σ→R
  return grad_R;
}

// [[Rcpp::export]]
double log_dlkjcorr(const arma::mat& R, 
                    double eta) {
  int d = R.n_rows;
  if (R.n_cols != d) throw std::runtime_error("Matrix R must be square");
  
  // Compute log determinant of R
  double sign = 0.0;
  double logdet = 0.0;
  arma::log_det(logdet, sign, R);
  
  if (sign <= 0.0) {
    throw std::runtime_error("Matrix R is not positive definite");
  }
  
  double log_const = 0.0;
  for (int k = 1; k <= d - 1; ++k) {
    int m = d - 1 - k;
    log_const += 2.0 * k * std::log(2.0);
    log_const += std::lgamma(eta + m / 2.0) - std::lgamma(eta + (d - 1) / 2.0);
  }
  return log_const;
  double log_density = log_const + (eta - 1.0) * logdet;
  return log_density;
}

// [[Rcpp::export]]
double log_dens_multi(const arma::uvec& y, 
                      const arma::mat& X, 
                      const arma::uvec& bt_id,
                      const arma::uvec& lg_id, 
                      const arma::mat& beta, 
                      const arma::mat& u,
                      int L){
  int N = y.n_rows;
  int P = beta.n_rows;
  int K = beta.n_cols;
  arma::mat log_odds = X * beta;
  double out = 0;
  for (int i = 0; i < N; ++i){
    log_odds.row(i) += u.row(L*bt_id(i)+lg_id(i));
    out += log_odds(i,y(i)) - log(sum(exp(log_odds.row(i))));
  }
  for (int k = 1; k < K; ++k){
    for (int p = 0; p < P; ++p){
      out += R::dnorm(beta(p,k),0.0,5.0,true);
    }
  }
  return out;
}

// [[Rcpp::export]]
double log_dens_ar1(const arma::mat& u,
                   const arma::mat& rho,
                   const arma::mat& tau,
                   const arma::cube& L_corr,
                   int B,
                   const arma::uvec& T_b){
  int L = rho.n_rows;
  int K = rho.n_cols + 1;
  double out = 0;
  for (int k = 1; k < K; ++k){
    // Obtain Innovation and Stationary Matrices
    arma::mat L_corr_k = L_corr.slice(k-1);
    arma::vec tau_k = tau.col(k-1);
    arma::vec rho_k = rho.col(k-1);
    arma::mat Sigma_0_k = arma::diagmat(tau_k) *
      L_corr_k * arma::diagmat(tau_k);
    for (int i = 0; i < L; ++i){
      for (int j = 0; j < L; ++j){
        Sigma_0_k(i,j) /= (1.0 - rho_k(i) * rho_k(j));
      }
    }
    arma::vec tau_0_k = arma::sqrt(Sigma_0_k.diag());
    arma::mat L_corr_0_k = arma::diagmat(1.0/tau_0_k) *
      Sigma_0_k *
      arma::diagmat(1.0/tau_0_k);
    int i = 0;
    for (int b = 0; b < B; ++b){
      for (int t = 0; t < T_b(b); ++t){
        if (t == 0){
          out += log_dmvnorm(u.col(k).rows(L*i,L*(i+1)-1),
                             arma::zeros<arma::vec>(L),
                             tau_0_k,
                             L_corr_0_k);
        } else {
          out += log_dmvnorm(u.col(k).rows(L*i,L*(i+1)-1),
                             rho_k % u.col(k).rows(L*(i-1),L*i-1),
                             tau_k,
                             L_corr_k);
        }
        ++i;
      }
    }
    out += log_dlkjcorr(L_corr.slice(k-1),0.5);
    for (int l = 0; l < L; ++l){
      out += R::dnorm(rho(l,k-1),1.0,1.0,true);
      out += R::dnorm(tau(l,k-1),0.0,1.0,true);
    }
  }
  return out;
}

// [[Rcpp::export]]
arma::mat dlog_dens_multi_dbeta(const arma::mat& y_mat, 
                                const arma::mat& X, 
                                const arma::uvec& bt_id,
                                const arma::uvec& lg_id,
                                const arma::mat& beta, 
                                const arma::mat& u,
                                int L){
  int K = beta.n_cols;
  int N = y_mat.n_rows;
  arma::mat odds = exp(X * beta);
  arma::mat prob(N,K);
  for (int i = 0; i < N; i++){
    odds.row(i) %= exp(u.row(L*bt_id(i)+lg_id(i)));
    prob.row(i) = odds.row(i) / sum(odds.row(i));
  }
  arma::mat out = X.t() * (y_mat - prob);
  out += -beta / 25.0;
  return out;
}

// [[Rcpp::export]]
arma::mat dlog_dens_ar1_dtau(const arma::mat& u,
                             const arma::mat& rho,
                             const arma::mat& tau,
                             const arma::cube& L_corr,
                             int B,
                             const arma::uvec& T_b){
  int L = rho.n_rows;
  int K = rho.n_cols + 1;
  arma::mat out(L,K-1,arma::fill::zeros);
  for (int k = 1; k < K; ++k){
    arma::vec rho_k = rho.col(k-1);
    arma::vec tau_k = tau.col(k-1);
    arma::mat Lcorr_k = L_corr.slice(k-1);
    arma::mat Sigma_k = arma::diagmat(tau_k) * Lcorr_k * arma::diagmat(tau_k);
    arma::mat Sigma0_k = Sigma_k;
    for(int i = 0; i < L; ++i)
      for(int j = 0; j < L; ++j)
        Sigma0_k(i,j) /= (1.0 - rho_k(i) * rho_k(j));
    
    int i = 0;
    for(int b = 0; b < B; ++b){
      for(int t = 0; t < T_b(b); ++t){
        arma::vec u_bt = u.col(k).rows(L*i,L*(i+1)-1);
        arma::vec delta;
        arma::mat Sigma_use;
        if(t == 0){
          delta = u_bt;
          Sigma_use = Sigma0_k;
        } else {
          arma::vec u_prev = u.col(k).rows(L*(i-1),L*i-1);
          delta = u_bt - rho_k % u_prev;
          Sigma_use = Sigma_k;
          // dlog_rho.col(k) += (Sigma_use.i() * delta) % u_prev; // elementwise
        }
        arma::mat temp = Sigma_use.i() * delta * delta.t() * Sigma_use.i() - Sigma_use.i();
        out.col(k-1) += temp.diag() % tau_k;  // chain through diag
        //dlog_Lcorr[k] = arma::diagmat(1.0/tau_k) * temp * arma::diagmat(1.0/tau_k); 
        ++i;
      }
    }
    // priors
    // dlog_rho.col(k) -= (rho_k - 1.0);
    out.col(k-1) -= tau_k;
    // LKJ derivative not implemented here
  }
  return out;
}

// [[Rcpp::export]]
arma::cube dlog_dens_ar1_dL_corr(const arma::mat& u,
                                  const arma::mat& rho,
                                  const arma::mat& tau,
                                  const arma::cube& L_corr,
                                  int B,
                                  const arma::uvec& T_b){
  int L = rho.n_rows;
  int K = rho.n_cols + 1;
  arma::cube out(L,L,K-1,arma::fill::zeros);
  for (int k = 1; k < K; ++k){
    arma::vec rho_k = rho.col(k-1);
    arma::vec tau_k = tau.col(k-1);
    arma::mat Lcorr_k = L_corr.slice(k-1);
    arma::mat Sigma_k = arma::diagmat(tau_k) * Lcorr_k * arma::diagmat(tau_k);
    arma::mat Sigma0_k = Sigma_k;
    for(int i = 0; i < L; ++i)
      for(int j = 0; j < L; ++j)
        Sigma0_k(i,j) /= (1.0 - rho_k(i) * rho_k(j));
    
    int i = 0;
    for(int b = 0; b < B; ++b){
      for(int t = 0; t < T_b(b); ++t){
        arma::vec u_bt = u.col(k).rows(L*i,L*(i+1)-1);
        arma::vec delta;
        arma::mat Sigma_use;
        if(t == 0){
          delta = u_bt;
          Sigma_use = Sigma0_k;
        } else {
          arma::vec u_prev = u.col(k).rows(L*(i-1),L*i-1);
          delta = u_bt - rho_k % u_prev;
          Sigma_use = Sigma_k;
          // dlog_rho.col(k) += (Sigma_use.i() * delta) % u_prev; // elementwise
        }
        arma::mat temp = Sigma_use.i() * delta * delta.t() * Sigma_use.i() - Sigma_use.i();
        //out.col(k) += temp.diag() % tau_k;  // chain through diag
        out.slice(k-1) += arma::diagmat(1.0/tau_k) * temp * arma::diagmat(1.0/tau_k); 
        ++i;
      }
    }
    out.slice(k-1) += -0.5 * arma::inv_sympd(Lcorr_k);
  }
  return out;
}

// [[Rcpp::export]]
arma::mat dlog_dens_ar1_drho(const arma::mat& u,
                              const arma::mat& rho,
                              const arma::mat& tau,
                              const arma::cube& L_corr,
                              int B,
                              const arma::uvec& T_b){
  int L = rho.n_rows;
  int K = rho.n_cols + 1;
  arma::mat out(L,K-1,arma::fill::zeros);
  for (int k = 1; k < K; ++k){
    arma::vec rho_k = rho.col(k-1);
    arma::vec tau_k = tau.col(k-1);
    arma::mat Lcorr_k = L_corr.slice(k-1);
    arma::mat Sigma_k = arma::diagmat(tau_k) * Lcorr_k * arma::diagmat(tau_k);
    arma::mat Sigma0_k = Sigma_k;
    for(int i = 0; i < L; ++i)
      for(int j = 0; j < L; ++j)
        Sigma0_k(i,j) /= (1.0 - rho_k(i) * rho_k(j));
    
    int i = 0;
    for(int b = 0; b < B; ++b){
      for(int t = 0; t < T_b(b); ++t){
        arma::vec u_bt = u.col(k).rows(L*i,L*(i+1)-1);
        arma::vec delta;
        arma::mat Sigma_use;
        if(t == 0){
          delta = u_bt;
          Sigma_use = Sigma0_k;
        } else {
          arma::vec u_prev = u.col(k).rows(L*(i-1),L*i-1);
          delta = u_bt - rho_k % u_prev;
          Sigma_use = Sigma_k;
          // dlog_rho.col(k) += (Sigma_use.i() * delta) % u_prev; // elementwise
        }
        arma::mat temp = Sigma_use.i() * delta * delta.t() * Sigma_use.i() - Sigma_use.i();
        out.col(k-1) += temp.diag() % tau_k;  // chain through diag
        //out.slice(k-1) = arma::diagmat(1.0/tau_k) * temp * arma::diagmat(1.0/tau_k); 
        ++i;
      }
    }
    out.col(k-1) += -(rho_k - 1);
  }
  return out;
}

// [[Rcpp::export]]
List update_beta(const arma::uvec& y,
                 const arma::mat& y_mat, 
                 const arma::mat& X, 
                 const arma::uvec& bt_id, 
                 const arma::uvec& lg_id, 
                 const arma::mat& beta, 
                 const arma::mat& u,
                 int L, int T,
                 double eps){
  int P = X.n_cols;
  int K = beta.n_cols;
  arma::mat prop(P,K,arma::fill::zeros);
  arma::mat dbeta_curr = dlog_dens_multi_dbeta(
    y_mat,X,bt_id,lg_id,beta,u,L
  );
  arma::mat mu_curr = beta.submat(0,1,P-1,K-1) +
    0.5 * pow(eps,2.0) * dbeta_curr.submat(0,1,P-1,K-1);
  prop.submat(0,1,P-1,K-1) = eps * arma::randn<arma::mat>(P,K-1) +
    mu_curr;
  arma::mat dbeta_prop = dlog_dens_multi_dbeta(
    y_mat,X,bt_id,lg_id,prop,u,L
  );
  arma::mat mu_prop = prop.submat(0,1,P-1,K-1) +
    0.5 * pow(eps,2.0) * dbeta_prop.submat(0,1,P-1,K-1);
  double log_post_prop = log_dens_multi(y,X,bt_id,lg_id,prop,u,L);
  double log_post_curr = log_dens_multi(y,X,bt_id,lg_id,beta,u,L);
  double log_q_curr = log_dnorm_vec(
    arma::vectorise(prop.submat(0,1,P-1,K-1)),
    arma::vectorise(mu_curr),
    eps
  );
  double log_q_prop = log_dnorm_vec(
    arma::vectorise(beta.submat(0,1,P-1,K-1)),
    arma::vectorise(mu_prop),
    eps
  );
  // Metropolis acceptance step
  double log_alpha = log_post_prop - log_q_curr - log_post_curr + log_q_prop;
  if (std::log(arma::randu()) >= log_alpha) {
    prop = beta;
  }
  return List::create(
    Named("beta") = prop,
    Named("alpha") = std::min(exp(log_alpha),1.0)
  );
}

// [[Rcpp::export]]
List update_rho(const arma::mat& u,
                const arma::mat& rho,
                const arma::mat& tau,
                const arma::cube& L_corr,
                int B,
                const arma::uvec& T_b,
                double eps){
  int L = rho.n_rows;
  int K = rho.n_cols + 1;
  arma::mat mu_curr = rho +
    0.5 * pow(eps,2.0) * dlog_dens_ar1_drho(
      u,rho,tau,L_corr,B,T_b
    );
  arma::mat prop = eps * arma::randn<arma::mat>(L,K-1) + 
    mu_curr;
  arma::mat mu_prop = prop +
    0.5 * pow(eps,2.0) * dlog_dens_ar1_drho(
      u,prop,tau,L_corr,B,T_b
    );
  double log_post_prop;
  if (arma::min(arma::min(prop)) < -1){
    log_post_prop = R_NegInf;
  } else if (arma::max(arma::max(prop)) > 1){
    log_post_prop = R_NegInf;
  } else {
    log_post_prop = log_dens_ar1(
      u,prop,tau,L_corr,B,T_b
    );
  }
  double log_post_curr = log_dens_ar1(
    u,rho,tau,L_corr,B,T_b
  );
  double log_q_curr = log_dnorm_vec(
    arma::vectorise(prop),
    arma::vectorise(mu_curr),
    eps
  );
  double log_q_prop = log_dnorm_vec(
    arma::vectorise(rho),
    arma::vectorise(mu_prop),
    eps
  );
  // Metropolis acceptance step
  double log_alpha = log_post_prop - log_q_curr - log_post_curr + log_q_prop;
  if (std::log(arma::randu()) >= log_alpha) {
    prop = rho;
  }
  return List::create(
    Named("rho") = prop,
    Named("alpha") = std::min(exp(log_alpha),1.0)
  );
}

// [[Rcpp::export]]
List update_tau(const arma::mat& u,
                const arma::mat& rho,
                const arma::mat& tau,
                const arma::cube& L_corr,
                int B,
                const arma::uvec& T_b,
                double eps){
  int L = rho.n_rows;
  int K = rho.n_cols + 1;
  arma::mat mu_curr = tau + 0.5 * pow(eps,2.0) * 
    dlog_dens_ar1_dtau(
      u,rho,tau,L_corr,B,T_b
    );
  arma::mat prop = eps * arma::randn<arma::mat>(L,K-1) +
    mu_curr;
  arma::mat mu_prop = prop + 0.5 * pow(eps,2.0) * 
    dlog_dens_ar1_dtau(
      u,rho,prop,L_corr,B,T_b
    );
  double log_post_prop;
  if (arma::min(arma::min(prop)) < 0){
    log_post_prop = R_NegInf;
  } else {
    log_post_prop = log_dens_ar1(
      u,rho,prop,L_corr,B,T_b
    );
  }
  double log_post_curr = log_dens_ar1(
    u,rho,tau,L_corr,B,T_b
  );
  double log_q_curr = log_dnorm_vec(
    arma::vectorise(prop),
    arma::vectorise(mu_curr),
    eps
  );
  double log_q_prop = log_dnorm_vec(
    arma::vectorise(tau),
    arma::vectorise(mu_prop),
    eps
  );
  // Metropolis acceptance step
  double log_alpha = log_post_prop - log_q_curr - log_post_curr + log_q_prop;
  if (std::log(arma::randu()) >= log_alpha) {
    prop = tau;
  }
  return List::create(
    Named("tau") = prop,
    Named("alpha") = std::min(exp(log_alpha),1.0)
  );
}

// [[Rcpp::export]]
List update_L_corr(const arma::mat& u,
                   const arma::mat& rho,
                   const arma::mat& tau,
                   const arma::cube& L_corr,
                   int B,
                   const arma::uvec& T_b,
                   double eps){
  int L = rho.n_rows;
  int K = rho.n_cols + 1;
  arma::cube prop(L,L,K-1);
  arma::cube mu_curr = L_corr + 0.5 * pow(eps,2.0) * 
    dlog_dens_ar1_dL_corr(
      u,rho,tau,L_corr,B,T_b
    );
  for (int k = 0; k < (K - 1); ++k){
    for (int l = 0; l < L; ++l){
      for (int lp = l; lp < L; ++lp){
        if (lp == l){
          prop(l,lp,k) = 1;
        } else {
          prop(l,lp,k) = eps * arma::randn() +
            mu_curr(l,lp,k);
          prop(lp,l,k) = prop(l,lp,k);
        }
      }
    }
  }
  arma::cube mu_prop = prop + 0.5 * pow(eps,2.0) * 
    dlog_dens_ar1_dL_corr(
      u,rho,tau,prop,B,T_b
    );
  arma::vec dets(prop.n_slices);
  
  for (size_t k = 0; k < prop.n_slices; ++k) {
    dets(k) = arma::det(prop.slice(k));
  }
  double log_post_prop;
  if (arma::min(dets) < 0){
    log_post_prop = R_NegInf;
  } else {
    log_post_prop = log_dens_ar1(
      u,rho,tau,prop,B,T_b
    );
  }
  double log_post_curr = log_dens_ar1(
    u,rho,tau,L_corr,B,T_b
  );
  double log_q_curr = 0;
  double log_q_prop = 0;
  for (int k = 1; k < K ; ++k){
    log_q_curr += log_dnorm_vec(
      arma::vectorise(arma::trimatl(prop.slice(k-1))),
      arma::vectorise(arma::trimatl(mu_curr.slice(k-1))),
      eps
    );
    log_q_prop += log_dnorm_vec(
      arma::vectorise(arma::trimatl(L_corr.slice(k-1))),
      arma::vectorise(arma::trimatl(mu_prop.slice(k-1))),
      eps
    );
  }
  // Metropolis acceptance step
  double log_alpha = log_post_prop - log_q_curr - log_post_curr + log_q_prop;
  if (std::log(arma::randu()) >= log_alpha) {
    prop = L_corr;
  }
  return List::create(
    Named("L_corr") = L_corr,
    Named("alpha") = std::min(exp(log_alpha),1.0)
  );
}

// [[Rcpp::export]]
arma::mat pgas_u(const arma::uvec& y,
                 const arma::mat& X,
                 const arma::uvec& bt_id,
                 const arma::uvec& lg_id,
                 const arma::mat& beta,
                 const arma::mat& u,
                 const arma::mat& rho,
                 const arma::mat& tau,
                 const arma::cube& L_corr,
                 int B,
                 const arma::uvec& T_b,
                 int N){
  int K = beta.n_cols;
  int L = rho.n_rows;
  int N_u = u.n_rows;

  arma::uvec idx;
  arma::uvec ancestors(N);
  arma::vec logancestorWeights(N);
  arma::vec logWeights(N);
  double maxlogW;
  arma::vec weights(N);
  arma::vec ancestorWeights(N);
  arma::mat out(N_u, K, arma::fill::zeros);
  arma::cube u_int(N_u, K, N, arma::fill::randn);
  arma::cube u_fin(N_u, K, N, arma::fill::zeros);
  u_int.slice(N-1) = u;
  u_int.col(0) = arma::zeros<arma::cube>(u_int.n_rows,1,u_int.n_slices);

  std::vector<arma::uvec> idx_z(N_u / L);
  for (int i = 0; i < (N_u / L); ++i) {
    idx_z[i] = arma::find(bt_id == i);
  }

  // Obtain Stationary Matrices
  std::vector<arma::mat> Sigma(K-1);
  std::vector<arma::mat> Sigma_0(K-1);
  for (int k = 1; k < K; ++k) {
    // Obtain Innovation and Stationary Matrices
    arma::mat L_corr_k = L_corr.slice(k-1);
    arma::vec tau_k = tau.col(k-1);
    arma::vec rho_k = rho.col(k-1);
    arma::mat Sigma_k = arma::diagmat(tau_k) *
      L_corr_k * arma::diagmat(tau_k);
    arma::mat Sigma_0_k = Sigma_k;
    for (int i = 0; i < L; ++i){
      for (int j = 0; j < L; ++j){
        Sigma_0_k(i,j) /= (1.0 - rho_k(i) * rho_k(j));
      }
    }
    Sigma[k-1] = Sigma_k;
    Sigma_0[k-1] = Sigma_0_k;
  }

  int j = 0;
  for (int b = 0; b < B; ++b){
    arma::uword b_start = L*j;
    arma::uword b_end = L*(j+T_b(b))-1;
    arma::uword row_start = L*j;
    arma::uword row_end   = L*(j+1)-1;
    for (int k = 1; k < K; ++k){
      for (int i = 0; i < (N-1); ++i){
        u_int.slice(i).submat(row_start,k,row_end,k) = 
          arma::chol(Sigma_0[k-1]) * 
          u_int.slice(i).submat(row_start,k,row_end,k);
      }
    }
    idx = idx_z[j];
    arma::uvec y_sub = y.elem(idx);
    arma::mat X_sub = X.rows(idx);
    arma::uvec bt_id_sub = bt_id(idx);
    arma::uvec lg_id_sub = lg_id(idx);
    for (int i = 0; i < N; ++i){
      logWeights(i) = log_dens_multi(y_sub,
                 X_sub,
                 bt_id_sub,
                 lg_id_sub,
                 beta,
                 u_int.slice(i),
                 L);
    }
    maxlogW = max(logWeights);
    weights = arma::normalise(exp(logWeights - maxlogW));
    ++j;
    if (T_b(b) > 1){
      for (int t = 1; t < T_b(b); ++t){
        arma::uword row_start = L*j;
        arma::uword row_end   = L*(j+1)-1;
        ancestors = systematic_resample(weights,N-1);
        for (int k = 1; k < K; ++k){
          for (int i = 0; i < (N-1); ++i){
            u_int.slice(i).submat(row_start,k,row_end,k) = 
              arma::chol(Sigma[k-1]) * 
              u_int.slice(i).submat(row_start,k,row_end,k) +
              rho.col(k-1) % u_int.slice(ancestors(i)).submat(row_start-L,k,row_end-L,k);
          }
        }
        // Ancestor sampling
        for (int i=0; i < N; ++i){
          logancestorWeights(i) = logWeights(i);
          for (int k = 1; k < K; ++k){
            logancestorWeights(i) += log_dmvnorm(u.submat(row_start,k,row_end,k),
                               rho.col(k-1) % u_int.slice(i).submat(row_start-L,k,row_end-L,k),
                               tau.col(k-1), 
                               L_corr.slice(k-1));
          }
        }
        ancestorWeights = arma::normalise(exp(logancestorWeights - max(logancestorWeights)));
        ancestors[N-1] = systematic_resample(ancestorWeights,1)[0];
        for (int i = 0; i < N; ++i){
          u_fin.subcube(b_start,0,i,row_end,K-1,i) = 
            u_int.subcube(b_start,0,ancestors[i],row_end,K-1,ancestors[i]);
        }
        u_int = u_fin;
        idx = idx_z[j];
        if (idx.n_elem > 0){
          arma::uvec y_sub = y.elem(idx);
          arma::mat X_sub = X.rows(idx);
          arma::uvec bt_id_sub = bt_id(idx);
          arma::uvec lg_id_sub = lg_id(idx);
          for (int i = 0; i < N; ++i){
            logWeights(i) = log_dens_multi(y_sub,
                       X_sub,
                       bt_id_sub,
                       lg_id_sub,
                       beta,
                       u_int.slice(i),
                       L);
          }
        } else {
          logWeights = arma::zeros<arma::vec>(N);
        }
        maxlogW = max(logWeights);
        weights = arma::normalise(exp(logWeights - maxlogW));
        ++j;
      }
    }
    int ind = systematic_resample(weights,1)[0];
    out.submat(b_start,0,b_end,K-1) = 
      u_fin.slice(ind).submat(b_start,0,b_end,K-1);
  }
  return out;
}





#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <RcppTN.h>

//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::depends(RcppTN)]]
using namespace Rcpp;

namespace lrf{
std::vector<int> csample_int( std::vector<int> x,
                              int size,
                              bool replace,
                              NumericVector prob) {
  std::vector<int> ret = RcppArmadillo::sample(x, size, replace, prob) ;
  return ret ;
}
}

// [[Rcpp::export]]
double dmvnorm(const arma::vec& x, const arma::vec& mu, const arma::mat& Sigma, bool logd = false) {
  int d = x.n_elem; // Dimensionality
  arma::vec diff = x - mu;
  arma::mat invSigma = arma::inv_sympd(Sigma); // Inverse of covariance matrix
  double detSigma = arma::det(Sigma); // Determinant of covariance matrix
  double exponent = -0.5 * arma::as_scalar(diff.t() * invSigma * diff); // Quadratic form
  double norm_const = std::pow(2.0 * M_PI, -0.5 * d) * std::pow(detSigma, -0.5); // Normalization constant
  
  double density = norm_const * std::exp(exponent);
  return logd ? std::log(density) : density; // Return log-density if requested
}

// [[Rcpp::export]]
double log_post_multi(arma::vec y, arma::mat X, arma::vec time, arma::vec bowler, arma::vec league, arma::mat beta, arma::cube u, arma::cube Sigma, arma::cube Sigma_0, arma::mat rho){
  int N = y.n_rows;
  int K = beta.n_cols;
  int L = max(league);
  int B = max(bowler);
  int T = u.n_slices;
  arma::mat log_odds = X * beta;
  double out = 0;
  for (int i = 0; i < N; i++){
    log_odds.row(i) += u.slice(time(i)-1).row(league(i)*(bowler(i)-1));
    out += log_odds(i,y(i)) - log(sum(exp(log_odds.row(i))));
  }
  for (int k = 0; k < (K-1); k++){
    for (int b = 0; b < B; b++){
      for (int t = 1; t < T; t++){
        out += dmvnorm(u.slice(t).submat(L*(b-1),k,L*(b-1)+L-1,k),
                       rho.col(k)*u.slice(t-1).submat(L*(b-1),k,L*(b-1)+L-1,k),
                       Sigma.slice(k),
                       true);
      }
      out += dmvnorm(u.slice(0).submat(L*(b-1),k,L*(b-1)+L-1,k),
                     arma::zeros<arma::vec>(L),
                     Sigma_0.slice(k),
                     true);
    }
  }
  return out;
}

// [[Rcpp::export]]
arma::mat dlog_post_multi_dbeta(arma::mat y, arma::mat X, arma::vec time, arma::vec bowler, arma::vec league, arma::mat beta, arma::cube u, arma::uvec sub_rows){
  int Dx = X.n_cols;
  arma::mat y_sub = y.rows(sub_rows);
  arma::mat X_sub = X.rows(sub_rows);
  arma::vec time_sub = time(sub_rows);
  arma::vec bowler_sub = bowler(sub_rows);
  arma::vec league_sub = league(sub_rows);
  int K = beta.n_cols;
  int N = y_sub.n_rows;
  arma::mat odds = exp(X_sub * beta);
  arma::mat prob(N,K);
  for (int i = 0; i < N; i++){
    odds.row(i) %= exp(u.slice(time_sub(i)-1).row(league_sub(i)*(bowler_sub(i)-1)));
    prob.row(i) = odds.row(i) / sum(odds.row(i));
  }
  arma::mat out = X_sub.t() * (y_sub - prob);
  return out;
}

// [[Rcpp::export]]
arma::cube dlog_post_multi_du(arma::mat y, arma::mat X, arma::vec time, arma::vec bowler, arma::vec league, arma::mat beta, arma::cube u, arma::cube Sigma, arma::cube Sigma_0, arma::mat rho, arma::uvec sub_rows){
  arma::mat y_sub = y.rows(sub_rows);
  arma::mat X_sub = X.rows(sub_rows);
  arma::vec time_sub = time(sub_rows);
  arma::vec bowler_sub = bowler(sub_rows);
  arma::vec league_sub = league(sub_rows);
  int L = max(league);
  int B = max(bowler);
  int T = u.n_slices;
  int K = beta.n_cols;
  int N = y_sub.n_rows;
  arma::mat odds = exp(X_sub * beta);
  arma::mat prob(N,K);
  arma::cube out(B*L,K,T,arma::fill::zeros);
  for (int i = 0; i < N; i++){
    odds.row(i) %= exp(u.slice(time_sub(i)-1).row(league_sub(i)*(bowler_sub(i)-1)));
    prob.row(i) = odds.row(i) / sum(odds.row(i));
    out.slice(time_sub(i)-1).row(league_sub(i)*(bowler_sub(i)-1)) += y_sub.row(i) - prob.row(i);
  }
  //std::cout << rho.col(k) % arma::inv(Sigma.slice(0)) * (u.slice(1).submat(0,0,4,0) - rho.col(0) % u.slice(0).submat(0,0,4,0)) << std::endl;
  for (int k = 0; k < (K - 1); k++){
    for (int b = 1; b <= B; b++){
      out.slice(0).submat(L*(b-1),k,L*(b-1)+L-1,k) += -arma::inv(Sigma_0.slice(k)) * u.slice(0).submat(L*(b-1),k,L*(b-1)+L-1,k);
      for (int t = 1; t < T; t++){
        out.slice(t-1).submat(L*(b-1),k,L*(b-1)+L-1,k) += rho.col(k) % (arma::inv(Sigma.slice(k)) * (u.slice(t).submat(L*(b-1),k,L*(b-1)+L-1,k) - rho.col(k) % u.slice(t-1).submat(L*(b-1),k,L*(b-1)+L-1,k)));
        out.slice(t).submat(L*(b-1),k,L*(b-1)+L-1,k) += -arma::inv(Sigma.slice(k)) * (u.slice(t).submat(L*(b-1),k,L*(b-1)+L-1,k) - rho.col(k) % u.slice(t-1).submat(L*(b-1),k,L*(b-1)+L-1,k));
      }
    }
  }
  return out;
}

// [[Rcpp::export]]
arma::mat dlog_post_multi_drho(arma::vec time, arma::vec bowler, arma::vec league, arma::cube u, arma::cube Sigma, arma::cube Sigma_0, arma::mat rho){
  int L = max(league);
  int B = max(bowler);
  int T = u.n_slices;
  int K = u.n_cols;
  
  arma::mat out(L,K-1,arma::fill::zeros);
  for (int k = 0; k < (K - 1); k++){
    for (int b = 1; b <= B; b++){
      for (int t = 1; t < T; t++){
        out.col(k) += u.slice(t-1).submat(L*(b-1),k,L*(b-1)+L-1,k) % (arma::inv(Sigma.slice(k)) * (u.slice(t).submat(L*(b-1),k,L*(b-1)+L-1,k) - rho.col(k) % u.slice(t-1).submat(L*(b-1),k,L*(b-1)+L-1,k)));
      }
    }
  }
  return out;
}

// [[Rcpp::export]]
arma::cube dlog_post_multi_dSigma(arma::vec time, arma::vec bowler, arma::vec league, arma::cube u, arma::cube Sigma, arma::cube Sigma_0, arma::mat rho){
  int L = max(league);
  int B = max(bowler);
  int T = u.n_slices;
  int K = u.n_cols;
  arma::vec tmp(L);
  arma::cube out(L,L,K-1,arma::fill::zeros);
  for (int k = 0; k < (K - 1); k++){
    for (int b = 1; b <= B; b++){
      for (int t = 1; t < T; t++){
        tmp = u.slice(t).submat(L*(b-1),k,L*(b-1)+L-1,k) - rho.col(k) % u.slice(t-1).submat(L*(b-1),k,L*(b-1)+L-1,k);
        out.slice(k) += -arma::inv(Sigma.slice(k)) + arma::inv(Sigma.slice(k)) * tmp * tmp.t() * arma::inv(Sigma.slice(k));
      }
    }
  }
  return 0.5 * out;
}

// [[Rcpp::export]]
arma::cube dlog_post_multi_dSigma_0(arma::vec time, arma::vec bowler, arma::vec league, arma::cube u, arma::cube Sigma, arma::cube Sigma_0, arma::mat rho){
  int L = max(league);
  int B = max(bowler);
  int T = u.n_slices;
  int K = u.n_cols;
  arma::vec tmp(L);
  arma::cube out(L,L,K-1,arma::fill::zeros);
  for (int k = 0; k < (K - 1); k++){
    for (int b = 1; b <= B; b++){
      tmp = u.slice(0).submat(L*(b-1),k,L*(b-1)+L-1,k);
      out.slice(k) += -arma::inv(Sigma_0.slice(k)) + arma::inv(Sigma_0.slice(k)) * tmp * tmp.t() * arma::inv(Sigma_0.slice(k));
    }
  }
  return 0.5 * out;
}

// [[Rcpp::export]]
Rcpp::List Do_One_Step(arma::mat y, arma::mat X, arma::vec time, arma::vec bowler, arma::vec league, arma::mat beta, arma::cube u, arma::cube Sigma, arma::cube Sigma_0, arma::mat rho, arma::uvec sub_rows, double a){
  int Dx = X.n_cols;
  arma::mat y_sub = y.rows(sub_rows);
  arma::mat X_sub = X.rows(sub_rows);
  arma::vec time_sub = time(sub_rows);
  arma::vec bowler_sub = bowler(sub_rows);
  arma::vec league_sub = league(sub_rows);
  int L = arma::max(league);
  int B = arma::max(bowler);
  int T = u.n_rows;
  int K = beta.n_cols;
  int N = y_sub.n_rows;
  arma::mat d_beta = dlog_post_multi_dbeta(y,X,time,bowler,league,beta,u,sub_rows);
  arma::cube d_u = dlog_post_multi_du(y,X,time,bowler,league,beta,u,Sigma,Sigma_0,rho,sub_rows);
  arma::mat d_rho = dlog_post_multi_drho(time,bowler,league,u,Sigma,Sigma_0,rho);
  arma::cube d_Sigma = dlog_post_multi_dSigma(time,bowler,league,u,Sigma,Sigma_0,rho);
  arma::cube d_Sigma_0 = dlog_post_multi_dSigma_0(time,bowler,league,u,Sigma,Sigma_0,rho);

  arma::mat prop_beta(Dx,K);
  arma::cube prop_u(B*L,K,T);
  arma::mat prop_rho(L,K-1);
  arma::cube prop_Sigma(L,L,K-1);
  arma::cube prop_Sigma_0(L,L,K-1);
  for (int k = 0; k < (K-1); k++){
    for (int d = 0; d < Dx; d++){
      prop_beta(d,k) = beta(d,k) + a*d_beta(d,k);
    }
    for (int t = 0; t < T; t++){
      for (int i = 0; i < B*L; i++){
        prop_u(i,k,t) = u(i,k,t) + a*d_u(i,k,t);
      }
    }
    // for (int l = 0; l < L; l++){
    //   prop_rho(l,k) = rho(l,k) + a*d_rho(l,k);
    //   for (int m = 0; m < L; m++){
    //     prop_Sigma(l,m,k) = Sigma(l,m,k) + a*d_Sigma(l,m,k);
    //     prop_Sigma_0(l,m,k) = Sigma_0(l,m,k) + a*d_Sigma_0(l,m,k);
    //   }
    // }
  }
  return Rcpp::List::create(Named("beta") = prop_beta,
                            Named("u") = prop_u,
                            Named("rho") = prop_rho,
                            Named("Sigma") = prop_Sigma,
                            Named("Sigma_0") = prop_Sigma_0);
}

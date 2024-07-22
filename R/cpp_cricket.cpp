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
double log_post_multi(arma::vec y, arma::mat X, arma::mat beta, arma::mat Z_bowl, arma::mat u_bowl, arma::mat Z_bat, arma::mat u_bat, arma::mat Z_run, arma::mat u_run, arma::vec ltau){
  int N = y.n_rows;
  int Dx = X.n_cols;
  int Dbowl = Z_bowl.n_cols;
  int Dbat = Z_bat.n_cols;
  int Drun = Z_run.n_cols;
  int K = beta.n_cols;
  arma::mat log_odds = X * beta + Z_bowl * u_bowl + Z_bat * u_bat + Z_run * u_run;
  double out = 0;
  for (int i = 0; i < N; i++){
    out += log_odds(i,y(i)) - log(sum(exp(log_odds.row(i))));
  }
  for (int k = 0; k < (K-1); k++){
    for (int d = 0; d < Dx; d++){
      out += R::dnorm(beta(d,k), 0, 10, true);
    }
    for (int d = 0; d < Dbowl; d++){
      out += R::dnorm(u_bowl(d,k),0,exp(ltau[0]),true);
    }
    for (int d = 0; d < Dbat; d++){
      out += R::dnorm(u_bat(d,k),0,exp(ltau[1]),true);
    }
    for (int d = 0; d < Drun; d++){
      out += R::dnorm(u_run(d,k),0,exp(ltau[2]),true);
    }
  }
  for (int k = 0; k < 2; k++){
    out += R::dnorm(ltau[k],0,10,true);
  }
  return out;
}

// [[Rcpp::export]]
arma::mat dlog_post_multi_dbeta(arma::mat y, arma::mat X, arma::mat beta, arma::mat Z_bowl, arma::mat u_bowl, arma::mat Z_bat, arma::mat u_bat, arma::mat Z_run, arma::mat u_run, arma::vec ltau, arma::uvec sub_rows){
  int Dx = X.n_cols;
  int K = beta.n_cols;
  arma::mat y_sub = y.rows(sub_rows);
  arma::mat X_sub = X.rows(sub_rows);
  arma::mat Z_bowl_sub = Z_bowl.rows(sub_rows);
  arma::mat Z_bat_sub = Z_bat.rows(sub_rows);
  arma::mat Z_run_sub = Z_run.rows(sub_rows);
  int N = y_sub.n_rows;
  arma::mat odds = exp(X_sub * beta + Z_bowl_sub * u_bowl + Z_bat_sub * u_bat + Z_run_sub * u_run);
  arma::mat prob(N,K);
  arma::mat out(Dx,K);
  for (int i = 0; i < N; i++){
    prob.row(i) = odds.row(i) / sum(odds.row(i));
  }
  out = X_sub.t() * (y_sub - prob);
  out += -beta/100;
  return out.cols(0,K-2);
}

// [[Rcpp::export]]
arma::mat dlog_post_multi_dbowl(arma::mat y, arma::mat X, arma::mat beta, arma::mat Z_bowl, arma::mat u_bowl, arma::mat Z_bat, arma::mat u_bat, arma::mat Z_run, arma::mat u_run, arma::vec ltau, arma::uvec sub_rows){
  int Dz = Z_bowl.n_cols;
  int K = beta.n_cols;
  arma::mat y_sub = y.rows(sub_rows);
  arma::mat X_sub = X.rows(sub_rows);
  arma::mat Z_bowl_sub = Z_bowl.rows(sub_rows);
  arma::mat Z_bat_sub = Z_bat.rows(sub_rows);
  arma::mat Z_run_sub = Z_run.rows(sub_rows);
  int N = y_sub.n_rows;
  arma::mat odds = exp(X_sub * beta + Z_bowl_sub * u_bowl + Z_bat_sub * u_bat + Z_run_sub * u_run);
  arma::mat prob(N,K);
  arma::mat out(Dz,K);
  for (int i = 0; i < N; i++){
    prob.row(i) = odds.row(i) / sum(odds.row(i));
  }
  out = Z_bowl_sub.t() * (y_sub - prob);
  out += -u_bowl/pow(exp(ltau[0]),2);
  return out.cols(0,K-2);
}

// [[Rcpp::export]]
arma::mat dlog_post_multi_dbat(arma::mat y, arma::mat X, arma::mat beta, arma::mat Z_bowl, arma::mat u_bowl, arma::mat Z_bat, arma::mat u_bat, arma::mat Z_run, arma::mat u_run, arma::vec ltau, arma::uvec sub_rows){
  int Dz = Z_bat.n_cols;
  int K = beta.n_cols;
  arma::mat y_sub = y.rows(sub_rows);
  arma::mat X_sub = X.rows(sub_rows);
  arma::mat Z_bowl_sub = Z_bowl.rows(sub_rows);
  arma::mat Z_bat_sub = Z_bat.rows(sub_rows);
  arma::mat Z_run_sub = Z_run.rows(sub_rows);
  int N = y_sub.n_rows;
  arma::mat odds = exp(X_sub * beta + Z_bowl_sub * u_bowl + Z_bat_sub * u_bat + Z_run_sub * u_run);
  arma::mat prob(N,K);
  arma::mat out(Dz,K);
  for (int i = 0; i < N; i++){
    prob.row(i) = odds.row(i) / sum(odds.row(i));
  }
  out = Z_bat_sub.t() * (y_sub - prob);
  out += -u_bat/pow(exp(ltau[1]),2);
  return out.cols(0,K-2);
}

// [[Rcpp::export]]
arma::mat dlog_post_multi_drun(arma::mat y, arma::mat X, arma::mat beta, arma::mat Z_bowl, arma::mat u_bowl, arma::mat Z_bat, arma::mat u_bat, arma::mat Z_run, arma::mat u_run, arma::vec ltau, arma::uvec sub_rows){
  int Dz = Z_run.n_cols;
  int K = beta.n_cols;
  arma::mat y_sub = y.rows(sub_rows);
  arma::mat X_sub = X.rows(sub_rows);
  arma::mat Z_bowl_sub = Z_bowl.rows(sub_rows);
  arma::mat Z_bat_sub = Z_bat.rows(sub_rows);
  arma::mat Z_run_sub = Z_run.rows(sub_rows);
  int N = y_sub.n_rows;
  arma::mat odds = exp(X_sub * beta + Z_bowl_sub * u_bowl + Z_bat_sub * u_bat + Z_run_sub * u_run);
  arma::mat prob(N,K);
  arma::mat out(Dz,K);
  for (int i = 0; i < N; i++){
    prob.row(i) = odds.row(i) / sum(odds.row(i));
  }
  out = Z_run_sub.t() * (y_sub - prob);
  out += -u_run/pow(exp(ltau[2]),2);
  return out.cols(0,K-2);
}

// [[Rcpp::export]]
arma::vec dlog_post_multi_dltau(arma::mat X, arma::mat beta, arma::mat Z_bowl, arma::mat u_bowl, arma::mat Z_bat, arma::mat u_bat, arma::mat Z_run, arma::mat u_run, arma::vec ltau){
  int Dbowl = Z_bowl.n_cols;
  int Dbat = Z_bat.n_cols;
  int Drun = Z_run.n_cols;
  int K = beta.n_cols;
  arma::vec out(3);
  out.zeros();
  for (int k = 0; k < K; k++){
    for (int d = 0; d < Dbowl; d++){
      out(0) += -1 + 1/pow(exp(ltau[0]),2)*u_bowl(d,k);
    }
    for (int d = 0; d < Dbat; d++){
      out(1) += -1 + 1/pow(exp(ltau[1]),2)*u_bat(d,k);
    }
    for (int d = 0; d < Drun; d++){
      out(2) += -1 + 1/pow(exp(ltau[2]),2)*u_run(d,k);
    }
  }
  out(0) += -(ltau[0] - 0)/100;
  out(1) += -(ltau[1] - 0)/100;
  out(2) += -(ltau[2] - 0)/100;
  return out;
}

// [[Rcpp::export]]
double get_q(arma::vec y, arma::mat y_mat, arma::mat X, arma::mat beta, arma::mat prop_beta, arma::mat Z_bowl, arma::mat u_bowl, arma::mat prop_bowl, arma::mat Z_bat, arma::mat u_bat, arma::mat prop_bat, arma::mat Z_run, arma::mat u_run, arma::mat prop_run, arma::vec ltau, arma::vec prop_ltau, arma::uvec sub_rows, double a){
  int Dx = X.n_cols;
  int Dbowl = Z_bowl.n_cols;
  int Dbat = Z_bat.n_cols;
  int Drun = Z_run.n_cols;
  int K = beta.n_cols;
  double q = 0;
  q += log_post_multi(y,X,prop_beta,Z_bowl,prop_bowl,Z_bat,prop_bat,Z_run,prop_run,prop_ltau);
  q += -log_post_multi(y,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau);
  arma::mat d_prop_beta = dlog_post_multi_dbeta(y_mat,X,prop_beta,Z_bowl,prop_bowl,Z_bat,prop_bat,Z_run,prop_run,prop_ltau,sub_rows);
  arma::mat d_prop_bowl = dlog_post_multi_dbowl(y_mat,X,prop_beta,Z_bowl,prop_bowl,Z_bat,prop_bat,Z_run,prop_run,prop_ltau,sub_rows);
  arma::mat d_prop_bat = dlog_post_multi_dbat(y_mat,X,prop_beta,Z_bowl,prop_bowl,Z_bat,prop_bat,Z_run,prop_run,prop_ltau,sub_rows);
  arma::mat d_prop_run = dlog_post_multi_drun(y_mat,X,prop_beta,Z_bowl,prop_bowl,Z_bat,prop_bat,Z_run,prop_run,prop_ltau,sub_rows);
  arma::vec d_prop_ltau = dlog_post_multi_dltau(X,prop_beta,Z_bowl,prop_bowl,Z_bat,prop_bat,Z_run,prop_run,prop_ltau);
  arma::mat d_beta = dlog_post_multi_dbeta(y_mat,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau,sub_rows);
  arma::mat d_bowl = dlog_post_multi_dbowl(y_mat,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau,sub_rows);
  arma::mat d_bat = dlog_post_multi_dbat(y_mat,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau,sub_rows);
  arma::mat d_run = dlog_post_multi_drun(y_mat,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau,sub_rows);
  arma::vec d_ltau = dlog_post_multi_dltau(X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau);
  for (int k = 0; k < (K-1); k++){
    for (int d = 0; d < Dx; d++){
      q += -R::dnorm(prop_beta(d,k),a*d_beta(d,k),sqrt(2*a),true);
      q += R::dnorm(beta(d,k),a*d_prop_beta(d,k),sqrt(2*a),true);
    }
    for (int d = 0; d < Dbowl; d++){
      q += -R::dnorm(prop_bowl(d,k),a*d_bowl(d,k),sqrt(2*a),true);
      q += R::dnorm(u_bowl(d,k),a*d_prop_bowl(d,k),sqrt(2*a),true);
    }
    for (int d = 0; d < Dbat; d++){
      q += -R::dnorm(prop_bat(d,k),a*d_bat(d,k),sqrt(2*a),true);
      q += R::dnorm(u_bat(d,k),a*d_prop_bat(d,k),sqrt(2*a),true);
    }
    for (int d = 0; d < Drun; d++){
      q += -R::dnorm(prop_run(d,k),a*d_run(d,k),sqrt(2*a),true);
      q += R::dnorm(u_run(d,k),a*d_prop_run(d,k),sqrt(2*a),true);
    }
  }
  for (int k = 0; k < 2; k++){
    q += -R::dnorm(prop_ltau(k),a*d_ltau(k),sqrt(2*a),true);
    q += R::dnorm(ltau(k),a*d_prop_ltau(k),sqrt(2*a),true);
  }
  return q;
}

// [[Rcpp::export]]
Rcpp::List MALA(arma::vec y, arma::mat y_mat, arma::mat X, arma::mat beta, arma::mat Z_bowl, arma::mat u_bowl, arma::mat Z_bat, arma::mat u_bat, arma::mat Z_run, arma::mat u_run, arma::vec ltau, arma::uvec sub_rows, double a){
  int Dx = X.n_cols;
  int Dbowl = Z_bowl.n_cols;
  int Dbat = Z_bat.n_cols;
  int Drun = Z_run.n_cols;
  int K = beta.n_cols;
  arma::mat d_beta = dlog_post_multi_dbeta(y_mat,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau,sub_rows);
  arma::mat d_bowl = dlog_post_multi_dbowl(y_mat,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau,sub_rows);
  arma::mat d_bat = dlog_post_multi_dbat(y_mat,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau,sub_rows);
  arma::mat d_run = dlog_post_multi_drun(y_mat,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau,sub_rows);
  arma::vec d_ltau = dlog_post_multi_dltau(X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau);
  
  arma::mat prop_beta(Dx,K);
  arma::mat prop_bowl(Dbowl,K);
  arma::mat prop_bat(Dbat,K);
  arma::mat prop_run(Drun,K);
  arma::vec prop_ltau(3);
  for (int k = 0; k < (K-1); k++){
    for (int d = 0; d < Dx; d++){
      prop_beta(d,k) = R::rnorm(beta(d,k) + a*d_beta(d,k),sqrt(2*a));
    }
    for (int d = 0; d < Dbowl; d++){
      prop_bowl(d,k) = R::rnorm(u_bowl(d,k) + a*d_bowl(d,k),sqrt(2*a));
    }
    for (int d = 0; d < Dbat; d++){
      prop_bat(d,k) = R::rnorm(u_bat(d,k) + a*d_bat(d,k),sqrt(2*a));
    }
    for (int d = 0; d < Drun; d++){
      prop_run(d,k) = R::rnorm(u_run(d,k) + a*d_run(d,k),sqrt(2*a));
    }
  }
  for (int k = 0; k < 3; k++){
    prop_ltau(k) = R::rnorm(ltau(k) + a*d_ltau(k),sqrt(2*a));
  }
  double q = get_q(y,y_mat,X,beta,prop_beta,Z_bowl,u_bowl,prop_bowl,Z_bat,u_bat,prop_bat,Z_run,u_run,prop_run,ltau,prop_ltau,sub_rows,a);
  if (q > log(R::runif(0,1))){
    beta = prop_beta;
    u_bowl = prop_bowl;
    u_bat = prop_bat;
    u_run = prop_run;
    ltau = prop_ltau;
  }
  return Rcpp::List::create(Named("beta") = beta,
                            Named("u_bowl") = u_bowl,
                            Named("u_bat") = u_bat,
                            Named("u_run") = u_run,
                            Named("ltau") = ltau,
                            Named("q") = q);
}

// [[Rcpp::export]]
Rcpp::List Do_One_Step(arma::vec y, arma::mat y_mat, arma::mat X, arma::mat beta, arma::mat Z_bowl, arma::mat u_bowl, arma::mat Z_bat, arma::mat u_bat, arma::mat Z_run, arma::mat u_run, arma::vec ltau, arma::uvec sub_rows, double a){
  int Dx = X.n_cols;
  int Dbowl = Z_bowl.n_cols;
  int Dbat = Z_bat.n_cols;
  int Drun = Z_run.n_cols;
  int K = beta.n_cols;
  arma::mat d_beta = dlog_post_multi_dbeta(y_mat,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau,sub_rows);
  arma::mat d_bowl = dlog_post_multi_dbowl(y_mat,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau,sub_rows);
  arma::mat d_bat = dlog_post_multi_dbat(y_mat,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau,sub_rows);
  arma::mat d_run = dlog_post_multi_drun(y_mat,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau,sub_rows);
  arma::vec d_ltau = dlog_post_multi_dltau(X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau);
  
  arma::mat prop_beta(Dx,K);
  arma::mat prop_bowl(Dbowl,K);
  arma::mat prop_bat(Dbat,K);
  arma::mat prop_run(Drun,K);
  arma::vec prop_ltau(3);
  for (int k = 0; k < (K-1); k++){
    for (int d = 0; d < Dx; d++){
      prop_beta(d,k) = beta(d,k) - a*d_beta(d,k);
    }
    for (int d = 0; d < Dbowl; d++){
      prop_bowl(d,k) = u_bowl(d,k) - a*d_bowl(d,k);
    }
    for (int d = 0; d < Dbat; d++){
      prop_bat(d,k) = u_bat(d,k) - a*d_bat(d,k);
    }
    for (int d = 0; d < Drun; d++){
      prop_run(d,k) = u_run(d,k) - a*d_run(d,k);
    }
  }
  for (int k = 0; k < 3; k++){
    prop_ltau(k) = ltau(k) - a*d_ltau(k);
  }
  double log_post_prop = log_post_multi(y,X,prop_beta,Z_bowl,prop_bowl,Z_bat,prop_bat,Z_run,prop_run,prop_ltau);
  double log_post_curr = log_post_multi(y,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau);
  // if (log_post_prop > log_post_curr){
  //   beta = prop_beta;
  //   u_bowl = prop_bowl;
  //   u_bat = prop_bat;
  //   u_run = prop_run;
  //   ltau = prop_ltau;
  // }
  return Rcpp::List::create(Named("beta") = prop_beta,
                            Named("u_bowl") = prop_bowl,
                            Named("u_bat") = prop_bat,
                            Named("u_run") = prop_run,
                            Named("ltau") = prop_ltau,
                            Named("log_post_star") = log_post_prop,
                            Named("log_post_curr") = log_post_curr);
}

// // [[Rcpp::export]]
// arma::mat update_beta_multi(arma::vec y, arma::mat X, arma::mat beta, arma::mat Z, arma::mat u, double tau, double a) {
//   int Dx = X.n_cols;
//   int K = beta.n_cols;
//   arma::mat proposal(Dx,K);
//   double q;
//   arma::mat beta_sav = beta;
//   proposal = beta;
//   for (int k = 0; k < (K-1); k++){
//     for (int d = 0; d < Dx; d++){
//       std::std::cout << "(" << d << "," << k << ")"<< std::std::endl;
//       q = 0;
//       proposal(d,k) = R::rnorm(0.5*a*dlog_post_multi_dbeta(y,X,beta_sav,Z,u,tau,d,k),sqrt(a));
//       std::std::cout << proposal(d,k) << std::std::endl;
//       q = log_post_multi(y,X,proposal,Z,u,tau);
//       q += -log_post_multi(y,X,beta_sav,Z,u,tau);
//       q += -R::dnorm(proposal(d,k),0.5*a*dlog_post_multi_dbeta(y,X,beta_sav,Z,u,tau,d,k),sqrt(a),true);
//       q += R::dnorm(beta_sav(d,k),0.5*a*dlog_post_multi_dbeta(y,X,proposal,Z,u,tau,d,k),sqrt(a),true);
//       if (q > log(R::runif(0,1))) {
//         beta_sav = proposal;
//       }
//       std::std::cout << beta_sav(d,k) << std::std::endl;
//     }
//   }
//   //Rstd::cout << proposal << std::std::endl;
//   return beta_sav;
// }
// 
// // [[Rcpp::export]]
// arma::mat update_u_multi(arma::vec y, arma::mat X, arma::mat beta, arma::mat Z, arma::mat u, double tau, double a) {
//   int Dz = Z.n_cols;
//   int K = beta.n_cols;
//   arma::mat proposal(Dz,K);
//   double q;
//   arma::mat u_sav = u;
//   proposal = u;
//   for (int k = 0; k < (K-1); k++){
//     for (int d = 0; d < Dz; d++){
//       q = 0;
//       proposal(d,k) = R::rnorm(0.5*a*dlog_post_multi_dbeta(y,X,beta,Z,u_sav,tau,d,k),sqrt(a));
//       q = log_post_multi(y,X,beta,Z,proposal,tau);
//       q += -log_post_multi(y,X,beta,Z,u_sav,tau);
//       q += -R::dnorm(proposal(d,k),0.5*a*dlog_post_multi_dbeta(y,X,beta,Z,u_sav,tau,d,k),sqrt(a),true);
//       q += R::dnorm(u_sav(d,k),0.5*a*dlog_post_multi_dbeta(y,X,beta,Z,proposal,tau,d,k),sqrt(a),true);
//       if (q > log(R::runif(0,1)))  {
//         u_sav = proposal;
//       }
//     }
//   }
//   //Rstd::cout << proposal << std::std::endl;
//   return u_sav;
// }
// 
// 
// // [[Rcpp::export]]
// double update_tau_multi(arma::vec y, arma::mat X, arma::mat beta, arma::mat Z, arma::mat u, double tau, double a) {
//   double q;
//   double final;
//   double proposal = R::rnorm(0.5*a*dlog_post_multi_dtau(y,X,beta,Z,u,tau),sqrt(a));
//   if (proposal < 0){
//     q = R_NegInf;
//   } else {
//     q = log_post_multi(y,X,beta,Z,u,proposal);
//     q += -log_post_multi(y,X,beta,Z,u,tau);
//     q += -R::dnorm(proposal,0.5*a*dlog_post_multi_dtau(y,X,beta,Z,u,tau),sqrt(a),true);
//     q += R::dnorm(tau,0.5*a*dlog_post_multi_dtau(y,X,beta,Z,u,proposal),sqrt(a),true);
//   }
//   if (q > log(R::runif(0,1)))  {
//     final = proposal;
//   } else {
//     final = tau;
//   }
//   //Rstd::cout << proposal << std::std::endl;
//   return final;
// }

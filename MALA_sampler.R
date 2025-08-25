library(Rcpp)
library(nnet)
library(truncnorm)
source("cricket_data_org.R")
sourceCpp("R/cpp_cricket_V3.cpp")

keeps <- list(
  beta = array(NA, dim = c(P,K,R)),
  rho = array(NA, dim = c(L,K-1,R)),
  tau = array(NA, dim = c(L,K-1,R)),
  L_corr = array(NA, dim = c(L,L,K-1,R)),
  u = array(NA, dim = c(N_u,K,R))
)

beta   <- coef(multinom(factor(y) ~ 0 + X)) %>%
  t %>%
  cbind(0,.)
rho    <- matrix(0.5,nrow=L,ncol=K-1)
tau    <- 1 * sqrt(1 - rho^2)
L_corr <- array(0,dim=c(L,L,K-1))
u      <- matrix(0,nrow=N_u,K)
for (k in 1:(K-1)){
  L_corr[,,k] <- diag(1,L)
}
eps_beta    <- list(x = log(0.01),
                 xbar = 0,
                 H    = 0)
eps_rho     <- list(x = log(0.01),
                 xbar = 0,
                 H    = 0)
eps_tau     <- list(x = log(0.01),
                 xbar = 0,
                 H    = 0)
eps_L_corr  <- list(x = log(0.01),
                 xbar = 0,
                 H    = 0)

for (r in 1:(W+R)){
  print(paste0("Updating u: Iteration ",r))
  u <- pgas_u(y,X,
              u_id,lg_id,
              beta,u,rho,tau,L_corr,
              B,T_b,10)
  if (r > W){
    keeps$u[,,r-W] <- u
  }
  
  if (r <= W){
    print(paste0("Updating beta: Iteration ",r))
    tmp  <- update_beta(y,y_mat,X,u_id,lg_id,beta,u,
                        L,T,exp(eps_beta$x))
    beta <- tmp$beta
    eps_beta <- dual_averaging(tmp$alpha,0.574,r,
                               eps_beta$x,
                               eps_beta$xbar,
                               eps_beta$H)
    
    print(paste0("Updating rho: Iteration ",r))
    tmp <- update_rho(u,rho,tau,L_corr,
                      B,T_b,exp(eps_rho$x))
    rho <- tmp$rho
    eps_rho <- dual_averaging(tmp$alpha,0.574,r,
                              eps_rho$x,
                              eps_rho$xbar,
                              eps_rho$H)
    
    print(paste0("Updating tau: Iteration ",r))
    tmp <- update_tau(u,rho,tau,L_corr,
                      B,T_b,exp(eps_tau$x))
    tau <- tmp$tau
    eps_tau <- dual_averaging(tmp$alpha,0.574,r,
                              eps_tau$x,
                              eps_tau$xbar,
                              eps_tau$H)
    
    print(paste0("Updating L_corr: Iteration ",r))
    tmp <- update_L_corr(u,rho,tau,L_corr,
                         B,T_b,exp(eps_L_corr$x))
    L_corr <- tmp$L_corr
    eps_tau <- dual_averaging(tmp$alpha,0.574,r,
                              eps_L_corr$x,
                              eps_L_corr$xbar,
                              eps_L_corr$H)
  } else {
    print(paste0("Updating beta: Iteration ",r))
    tmp  <- update_beta(y,y_mat,X,u_id,lg_id,beta,u,
                        L,T,exp(eps_beta$xbar))
    beta <- tmp$beta
    keeps$beta[,,r-W] <- beta
    
    print(paste0("Updating rho: Iteration ",r))
    tmp <- update_rho(u,rho,tau,L_corr,
                      B,T_b,exp(eps_rho$xbar))
    rho <- tmp$rho
    keeps$rho[,,r-W] <- rho
    
    print(paste0("Updating tau: Iteration ",r))
    tmp <- update_tau(u,rho,tau,L_corr,
                      B,T_b,exp(eps_tau$xbar))
    tau <- tmp$tau
    keeps$tau[,,r-W] <- tau
    
    print(paste0("Updating L_corr: Iteration ",r))
    tmp <- update_L_corr(u,rho,tau,L_corr,
                         B,T_b,exp(eps_L_corr$xbar))
    L_corr <- tmp$L_corr
    keeps$L_corr[,,,r-W] <- L_corr
  }
}

saveRDS(keeps,"cricket_keeps.RDS")

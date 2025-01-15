#PARAMETERS
beta    <- matrix(0,nrow=P0,ncol=K)
u       <- array(0,dim = c(B*L,K,T))
rho     <- matrix(0,nrow=L,ncol=K-1)
Sigma   <- array(0,dim = c(L,L,K-1))
Sigma_0 <- array(0,dim = c(L,L,K-1))
for (k in 1:(K-1)){
  Sigma[,,k] <- diag(L)
  Sigma_0[,,k] <- diag(L)
}
a <- 1e-8

sourceCpp("cpp_cricket.cpp")

for (i in 1:10000){
  print(i)
  prop_beta    <- beta + a*dlog_post_multi_dbeta(y_mat,X,
                                                 time,bowler,league,
                                                 beta,u,
                                                 0)
  prop_u       <- u + a*dlog_post_multi_du(y_mat,X,
                                           time,bowler,league,
                                           beta,u,Sigma,Sigma_0,rho,
                                           0)
  prop_rho     <- rho + a*dlog_post_multi_drho(time,bowler,league,
                                               u,Sigma,Sigma_0,rho)
  prop_Sigma   <- Sigma + a*dlog_post_multi_dSigma(time,bowler,league,
                                                   u,Sigma,Sigma_0,rho)
  prop_Sigma_0 <- Sigma_0 + a*dlog_post_multi_dSigma_0(time,bowler,league,
                                                       u,Sigma,Sigma_0,rho)
  
  beta    <- prop_beta
  u       <- prop_u
  rho     <- prop_rho
  Sigma   <- prop_Sigma
  Sigma_0 <- prop_Sigma_0
  
  beta[,K] <- 0
  u[,K,]   <- 0
}


#PARAMETERS
#keeps is a list that stores kept draws (everything after burnin B)
keeps <- list(
  u_bowl    = array(NA, dim = c(R*n_chns/thin, Dbowl, K)),
  u_bat     = array(NA, dim = c(R*n_chns/thin, Dbat, K)),
  u_run     = array(NA, dim = c(R*n_chns/thin, Drun, K)),
  beta      = array(NA, dim = c(R*n_chns/thin, Dx, K)),
  ltau      = array(NA, dim = c(R*n_chns/thin, 3))
)

beta   <- matrix(0,nrow=Dx,ncol=K)
u_bowl <- matrix(0,nrow=Dbowl,ncol=K)
u_bat <- matrix(0,nrow=Dbat,ncol=K)
u_run <- matrix(0,nrow=Drun,ncol=K)
ltau   <- rep(0,3)
a      <- a0 <- 1e-8

H <- 0
delta <- 0.65
t0 <- 10
a_bar <- 1
gamma <- 0.05
kappa <- 0.75

#for (chn in 1:n_chns){
  for (i in 1:(R+B)){
    print(i)
    iter   <- MALA(y,y_mat,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau,a)
    beta   <- iter$beta
    u_bowl <- iter$u_bowl
    u_bat  <- iter$u_bat
    u_run  <- iter$u_run
    ltau <- iter$ltau
    if (i <= B){
      alpha <- exp(min(iter$q,0))
      # print(alpha)
      H <- (1 - 1/(i + t0))*H + 1/(i + t0)*(delta - alpha)
      a <- 10*a0*exp(-sqrt(i)/gamma*H)
      a_bar <- exp(i^(-kappa)*log(a) + (1 - i^(-kappa))*log(a_bar))
    }
    else {
      if (i %% thin == 0){
        #j = (R*(chn - 1) + i)/thin
        j = (R*(chn - 1) + i - B)/thin
        print(paste0("Saving Iteration: ",j))
        keeps$u_bowl[j,,] <- u_bowl
        keeps$u_bat[j,,]  <- u_bat
        keeps$u_run[j,,]  <- u_run
        keeps$beta[j,,]   <- beta
        keeps$ltau[j,]    <- ltau
      }
    }
  }
  #store after burn in
#}

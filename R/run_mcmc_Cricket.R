#PARAMETERS
#keeps is a list that stores kept draws (everything after burnin B)
# keeps <- list(
#   u_bowl    = array(NA, dim = c(R*n_chns/thin, Dbowl, K)),
#   u_bat     = array(NA, dim = c(R*n_chns/thin, Dbat, K)),
#   u_run     = array(NA, dim = c(R*n_chns/thin, Drun, K)),
#   beta      = array(NA, dim = c(R*n_chns/thin, Dx, K)),
#   ltau      = array(NA, dim = c(R*n_chns/thin, 3))
# )

beta   <- matrix(0,nrow=Dx,ncol=K)
u_bowl <- matrix(0,nrow=Dbowl,ncol=K)
u_bat <- matrix(0,nrow=Dbat,ncol=K)
u_run <- matrix(0,nrow=Drun,ncol=K)
ltau   <- rep(0,3)
a      <- a0 <- 1e-7

H <- 0
delta <- 0.65 
t0 <- 10
a_bar <- 1
gamma <- 0.05
kappa <- 0.75

tmp <- sample(0:(nrow(Cricket)-1))
M <- 5
cut_off <- (nrow(Cricket)/M) %>% floor()
remain <- nrow(Cricket) %% M
rand_samp <- list(tmp[1:cut_off],
                  tmp[(cut_off+1):(2*cut_off)],
                  tmp[(2*cut_off+1):(3*cut_off)],
                  tmp[(3*cut_off+1):(4*cut_off)],
                  tmp[(4*cut_off+1):(5*cut_off+1)])
curr_ll <- -Inf
prop_ll <- log_post_multi(y,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau)

# for (i in 1:(R+B)){
while(prop_ll - curr_ll > 1e-4){
  print(prop_ll)
  # if (i %% 5 == 0){
  #   sub_rows = 5*(0:((nrow(Cricket)/5) %>% floor()))
  # } else {
  #   sub_rows = 5*(1:((nrow(Cricket)/5) %>% floor()) - 1) + (i %% 5)
  # }
  # sub_rows = rand_samp[[i %% M + 1]]
  sub_rows = 0:(nrow(Cricket)-1)
  # iter   <- MALA(y,y_mat,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau,sub_rows,a)
  iter   <- Do_One_Step(y,y_mat,X,beta,Z_bowl,u_bowl,Z_bat,u_bat,Z_run,u_run,ltau,sub_rows,a)
  beta   <- iter$beta
  u_bowl <- iter$u_bowl
  u_bat  <- iter$u_bat
  u_run  <- iter$u_run
  ltau <- iter$ltau
  curr_ll <- iter$log_post_curr
  prop_ll <- iter$log_post_star
  # if (i <= B){
  #   alpha <- exp(min(iter$q,0))
  #   # print(alpha)
  #   H <- (1 - 1/(i + t0))*H + 1/(i + t0)*(delta - alpha)
  #   a <- 10*a0*exp(-sqrt(i)/gamma*H)
  #   a_bar <- exp(i^(-kappa)*log(a) + (1 - i^(-kappa))*log(a_bar))
  # } else {
  #   if (i %% thin == 0){
  #     #j = (R*(chn - 1) + i)/thin
  #     j = (R*(chn - 1) + i - B)/thin
  #     print(paste0("Saving Iteration: ",j))
  #     keeps$u_bowl[j,,] <- u_bowl
  #     keeps$u_bat[j,,]  <- u_bat
  #     keeps$u_run[j,,]  <- u_run
  #     keeps$beta[j,,]   <- beta
  #     keeps$ltau[j,]    <- ltau
  #   }
  # }
}


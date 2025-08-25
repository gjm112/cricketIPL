rm(list = ls())
source("cricket_data_org.R")

stan_mod <- stan_model("cricket_stan.stan")

fit <- sampling(stan_mod,
                list(N=N,P=P,B=B,L=L,K=K,
                     X=X,y=y,N_u=N_u,
                     u_id=u_id,lg_id=lg_id,T_b=T_b),
                iter=R+W,chains=1,warmup=W,
                pars = c("beta","rho","tau","L_corr_chol","u"),
                include = TRUE)  # Only include these parameters in the output
# Extract relevant draws only (e.g., beta and maybe rho)
draws <- as.data.frame(rstan::extract(fit))
saveRDS(draws,"MCMC_draws.RDS")

rm(list = ls())
source("cricket_data_org.R")

lg_id   <- as.numeric(factor(bbb$league))
bt_id   <- as.numeric(factor(bbb$striker))
sn_id   <- as.numeric(factor(bbb$year))

# Group the batters into S roughly equal shards
B_table <- bbb %>%
  group_by(striker) %>%
  summarise(T_b = length(unique(year)),
            n = n()) %>%
  ungroup()

BTL_table <- bbb %>% 
  group_by(striker,league,year) %>% 
  tally() %>%
  arrange(striker,year,league) %>%
  mutate(n = 1) %>%
  ungroup() %>%
  left_join(B_table %>% select(striker))

BT_table <- BTL_table %>% 
  arrange(league) %>%
  pivot_wider(names_from = league, 
              values_from = n,
              values_fill = 0) %>%
  arrange(striker,year)

stan_mod <- stan_model("cricket_stan_V2.stan")
  
u_idx_b <- as.numeric(factor(BTL_table$striker))
u_idx_l <- as.numeric(factor(BTL_table$league))
u_idx_t <- as.numeric(factor(BTL_table$year))
T_b     <- B_table$T_b
N_u     <- nrow(BTL_table)
ind_l_bt <- BT_table %>% select(-c(striker,year))
idx_l_bt_list <- apply(ind_l_bt, 1, function(row) which(row == 1))

# Step 2: Compute n_l_bt (number of leagues per row)
n_l_bt <- sapply(idx_l_bt_list, length)

# Step 3: Pad each row's index vector to length L (e.g., with 0s)
idx_l_bt <- t(sapply(idx_l_bt_list, function(idxs) {
  c(idxs, rep(0, L - length(idxs)))
}))

t_b <- as.numeric(factor(BT_table$year))
N_bt <- nrow(BT_table)

idx_z <- matrix(0, nrow = N_bt, ncol = L)
counter <- 1
for (i in 1:N_bt){
  for (j in 1:ncol(idx_l_bt)){
    if (idx_l_bt[i,j] != 0){
      idx_z[i,j] <- counter
      counter <- counter + 1
    }
  }
}

u_index <- integer(N)
for (n in 1:N) {
  b <- bt_id[n]
  l <- lg_id[n]
  t <- sn_id[n]
  u_index[n] <- which(u_idx_b == b & u_idx_l == l & u_idx_t == t)
}
# Fit model 
fit <- sampling(stan_mod,
                list(N=N,P=P,B=B,L=L,K=K,
                     X=X,y=y,N_u=N_u,u_index=u_index,
                     T_b=T_b,N_bt=N_bt,t_b=t_b,
                     n_l_bt=n_l_bt,idx_l_bt=idx_l_bt,idx_z=idx_z),
                iter=R+W,chains=1,warmup=W,
                pars = c("beta","rho","tau","L_corr_chol","u"),
                include = TRUE)  # Only include these parameters in the output
# Extract relevant draws only (e.g., beta and maybe rho)
draws <- as.data.frame(rstan::extract(fit))

# Save compressed RDS (or CSV if needed)
saveRDS(draws, file = "keeps.RDS")


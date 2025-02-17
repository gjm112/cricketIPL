rm(list=ls())
library(dplyr)
library(Rcpp)
library(rstan)
library(lubridate)
# library(LaplacesDemon)
# library(stats4)
# library(HMMpa)
library(nnet)
library(cricketdata)
library(lme4)
# library(mixcat)
library(glmm)

# get leagues individually bbb
# indian premier league
ipl_bbb = fetch_cricsheet('bbb', 'male', 'ipl') %>%
  mutate(league = 'IPL')

# big bash league australia
bbl_bbb = fetch_cricsheet('bbb', 'male', 'bbl') %>%
  mutate(league = 'BBL')

# pakistan super league
psl_bbb = fetch_cricsheet('bbb', 'male', 'psl') %>%
  mutate(league = 'PSL')

# caribbean premier league
cpl_bbb = fetch_cricsheet('bbb', 'male', 'cpl') %>%
  mutate(league = 'CPL')

# south african league
sat_bbb = fetch_cricsheet('bbb', 'male', 'sat')%>%
  mutate(league = 'SAT')

bbb = rbind(ipl_bbb, bbl_bbb, psl_bbb, cpl_bbb, sat_bbb)
names(bbb)

bbb <- bbb %>% 
  mutate(year = as.numeric(substr(season,1,4))) %>%
  filter(innings <= 2)

write.csv(bbb,"../data/cricket_data.csv",row.names = FALSE)

# model <- "
# data {
#   int<lower=2> K;                 // Number of categories
#   int<lower=1> N;                 // Number of observations
#   int<lower=1> P0;                // Number of fixed effects
#   int<lower=1> T;                 // Number of seasons
#   int<lower=1> L;                 // Number of leagues
#   int<lower=1> B;                 // Number of batters
#   int<lower=1,upper=K> y[N];      // Categorical outcome
#   int<lower=1> time[N];           // Season indicator
#   matrix[N, P0] X;                // Fixed effect model matrix
#   int<lower=1> bowler[N];         // Bowler indicator
#   int<lower=1> league[N];         // League indicator
# }
# parameters {
#   vector[P0] beta[K-1];                   // Fixed effects
#   matrix[T,B*L] u[K-1];                   // Random effects
#   vector<lower=-1, upper=1>[L] rho[K-1];  // AR(1) coefficients (diagonal of A)
#   cholesky_factor_corr[L] L_Omega[K-1];   // Cholesky factor of correlation matrix for categories
#   vector<lower=0>[L] sigma[K-1];          // Standard deviations for noise
# }
# model {
#   // Priors
#   for (k in 1:(K-1)){
#     beta[k] ~ normal(0,5);                // Prior on fixed effects
#     rho[k] ~ normal(0,5);                 // Prior on AR(1) coefficients
#     sigma[k] ~ normal(0,5);               // Prior on standard deviations
#     L_Omega[k] ~ lkj_corr_cholesky(1);    // LKJ prior for correlations
#   }
#   
#   // Hierarchical Structure
#   for (k in 1:(K-1)){
#     for (b in 1:B){
#       for (t in 2:T){
#         u[k][t,(L*(b-1)+1):(L*(b-1)+L)] ~
#           multi_normal_cholesky(
#             to_row_vector(rho[k]) .* u[k][t-1,(L*(b-1)+1):(L*(b-1)+L)],
#             diag_pre_multiply(
#               sigma[k],L_Omega[k]
#             )
#           );
#       }
#       u[k][1,(L*(b-1)+1):(L*(b-1)+L)] ~          
#           multi_normal_cholesky(
#             rep_vector(0,L),
#             diag_pre_multiply(
#               sigma[k] ./ sqrt((1 - pow(rho[k],2))),L_Omega[k]
#             )
#           );
#     }
#   }
# 
#   // Likelihood
#   for (n in 1:N) {
#     vector[K] logits;
#     for (k in 1:(K-1)){
#       logits[k] = X[n] * beta[k] + u[k][time[n],league[n]*(bowler[n]-1)+1];
#     }
#     logits[K] = 0; // Reference category
#     y[n] ~ categorical_logit(logits);
#   }
# }"
# 
# stan_mod <- stan_model(model_code = model)

X <- model.matrix(~ I(innings == 1) +
                    balls_remaining +
                    I(innings*target - target) +
                    runs_scored_yet +
                    wickets_lost_yet +
                    venue,
                  bbb)

league <- as.numeric(factor(bbb$league))
bowler <- as.numeric(factor(bbb$bowler))
y <- bbb$runs_off_bat
time <- bbb$time
y[y==4] <- 3
y[y==6] <- 4
y <- y + 1
y_mat   <- model.matrix(~0+as.character(y))
N <- length(y)
K <- length(unique(y))
P0 <- ncol(X)
B <- length(unique(bbb$bowler))
L <- length(unique(bbb$league))
T <- max(time)

source("run_mcmc_Cricket.R")

# fit <- sampling(stan_mod,
#                 list(K=K,N=N,P0=P0,
#                      y=y,X=X,
#                      bowler=bowler,league=league,time=time,
#                      B=B,L=L,T=T),
#                 iter=3000,chains=1,warmup=1000,
#                 pars = c("beta","u","rho","sigma","L_Omega"),
#                 control=list(max_treedepth=4))

library(tidyverse)
library(rstan)

last_two_digits <- function(x){
  substr(x,nchar(x)-1,nchar(x))
}

bbb <- read_csv("data/cricket_data.csv")
bbb <- bbb %>% 
  filter(year >= 2011)

X <- model.matrix(~ I(innings == 1) +
                    balls_remaining +
                    I(innings*target - target) +
                    runs_scored_yet +
                    wickets_lost_yet +
                    venue,
                  bbb)

league_id <- as.numeric(factor(bbb$league))
batter_id <- as.numeric(factor(bbb$striker))
y <- bbb$runs_off_bat
season_id <- as.numeric(factor(bbb$year))
y[y==4] <- 3
y[y==6] <- 4
y <- y + 1
N <- length(y)
K <- length(unique(y))
P <- ncol(X)
B <- length(unique(bbb$bowler))
L <- length(unique(bbb$league))
T <- length(unique(bbb$year))

stan_mod <- stan_model("cricket_stan.stan")
fit <- sampling(stan_mod,
                list(N=N,P=P,K=K,B=B,L=L,T=T,X=X,
                     batter_id=batter_id,
                     league_id=league_id,
                     season_id=season_id,
                     y=y),
                iter=5000,chains=1,warmup=1000,
                pars = c("beta","rho","L_k","u"),
                control=list(max_treedepth=5))


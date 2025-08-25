rm(list = ls())
library(tidyverse)
library(rstan)

# Load and Filter Data
bbb <- read_csv("data/cricket_data.csv")
bbb <- bbb %>% 
  filter(year >= 2011 & runs_off_bat !=3 & runs_off_bat != 5) %>%
  arrange(year)

# Extract X and y
X <- model.matrix(~ I(innings == 1) +
                    scale(balls_remaining) +
                    I(innings*scale(target) - scale(target)) +
                    scale(runs_scored_yet) +
                    scale(wickets_lost_yet) +
                    venue,
                  bbb)
y <- bbb$runs_off_bat
y[y==4] <- 3
y[y==6] <- 4
# y <- y + 1
y_mat <- model.matrix(~0+as.factor(y))
N <- nrow(bbb)
K <- length(unique(y))
P <- ncol(X)
B <- length(unique(bbb$striker))
L <- length(unique(bbb$league))
# T <- length(unique(bbb$year))
R <- 4000
W <- 1000
# S <- 100

bt_id <- as.numeric(factor(bbb$striker))
lg_id <- as.numeric(factor(bbb$league)) - 1
sn_id <- as.numeric(factor(bbb$year))

B_table <- bbb %>%
  group_by(striker) %>%
  summarise(min_yr = min(year),
            max_yr = max(year)) %>%
  ungroup() %>%
  rowwise() %>%
  mutate(year = list(seq(min_yr, max_yr))) %>%
  unnest(year) %>%
  select(striker,year)

u_idx_b <- as.numeric(factor(B_table$striker))
u_idx_t <- as.numeric(factor(B_table$year))
T_b     <- B_table %>%
  group_by(striker) %>%
  tally() %>%
  pull(n)
N_u     <- sum(T_b) * L

u_id <- integer(N)
for (n in 1:N) {
  b <- bt_id[n]
  t <- sn_id[n]
  u_id[n] <- which(u_idx_b == b & u_idx_t == t) - 1
}


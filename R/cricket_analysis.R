rm(list=ls())
library(tidyverse)
library(Rcpp)
# library(LaplacesDemon)
# library(stats4)
# library(HMMpa)
# library(nnet)
# library(lme4)

Cricket <- read.csv("../data/IPL_data.csv")

Cricket <- Cricket %>% 
  filter(runs_off_bat != 3 & runs_off_bat != 5)

X <- model.matrix(~ I(innings == 1) + 
                    balls_remaining + 
                    I(innings*target - target) + 
                    runs_scored_yet + 
                    wickets_lost_yet +
                    venue,
                  Cricket)
Z_bowl <- model.matrix(~ 0 + bowler,Cricket)
Z_bat <- model.matrix(~ 0 + striker,Cricket)
Z_run <- model.matrix(~ 0 + non_striker,Cricket)
y <- Cricket$runs_off_bat
y[y==4] <- 3
y[y==6] <- 4
y_mat <- model.matrix(~ 0 + as.character(y))
N <- length(y)
K <- length(unique(y))
Dx <- ncol(X)
Dbowl <- ncol(Z_bowl)
Dbat <- ncol(Z_bat)
Drun <- ncol(Z_run)

sourceCpp("cpp_cricket.cpp")
B <- 2000
R <- 20000
n_chns <- 1
thin <- 5
source("run_mcmc_cricket.R")
saveRDS(keeps,"Cricket_Post_Saves.RDS")

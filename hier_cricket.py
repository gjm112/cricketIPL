# Need to Change the Python directory using
# reticulate::use_python("C:/Users/mstuart1/OneDrive - Loyola University Chicago/Documents/.virtualenvs/r-reticulate/Scripts/python.exe")

import pandas as pd
import numpy as np
import math
import itertools
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset  
import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

## model
# Define a hierarchical model
class CategoricalHier(nn.Module):
    def __init__(self, n_fixed, B, L, T, K):
        # n_fixed: number of fixed effects
        # B, L, T: number of bowlers, leagues, years
        # K: number of possible runs off ball
        super().__init__()
        self.beta = nn.Linear(n_fixed, K)  # fixed effects
        self.u = nn.Embedding(B*L*T, K)  # random effects
        # Store K covariance matrices using Cholesky decomposition
        self.L_matrices = nn.ParameterList([
            nn.Parameter(torch.tril(torch.randn(L, L))) for _ in range(K)
        ])
        # Store K ar1 parameters for each league
        self.ar1 = nn.Parameter(torch.zeros(L, K))
        
    # Get logits to apply to loss function
    def forward(self, X, group_ids):
        # group_ids is an id from 0 to B*L*T - 1 (one for each bowler, league, year combination)
        fixed = self.beta(X)  # Linear transformation
        random = self.u(group_ids)  # Lookup random effects
        logits = fixed + random  # Sum fixed & random effects
        return logits  # Softmax applied in CrossEntropyLoss
    
    # Loss function for the random effects
    def blup_loss(B, L, T, K):
        u = self.u.weight # Extract the random effects shape (B*L*T,K)
        loss = 0
        
        for k in range(K):
            # Extract covariance matrix for k runs
            cov_matrix = self.L_matrices[k] @ self.L_matrices[k].T  # Construct full covariance
            for b in range(B):
                for t in range(1,T):
                    # Get the means for the AR1 process
                    mean_vector = self.ar1[:,k] * u[(L*T*b + L*(t-1)):(L*T*b + L*t),k]
                    # Multivariate normal loss function
                    mvn_dist = dist.MultivariateNormal(mean_vector,cov_matrix)
                    # Add the losses
                    loss += -torch.sum(mvn_dist.log_prob(u[(L*T*b + L*t):(L*T*b + L*(t+1)),k]))
        return loss

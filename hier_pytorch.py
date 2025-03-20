# Need to Change the Python directory using
# reticulate::use_python("C:/Users/mstuart1/OneDrive - Loyola University Chicago/Documents/.virtualenvs/r-reticulate/Scripts/python.exe")

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import cleaned_data as data

class HierarchicalMultinomialRegression(nn.Module):
    def __init__(self, n_fixed, B, L, T, K):
        super().__init__()
        self.n_fixed = n_fixed # Number of fixed effects
        self.B = B # Number of batters
        self.L = L # Number of leagues
        self.T = T # Number of seasons
        self.K = K # Number of runs

        # Fixed effects coefficients (Beta)
        self.beta = nn.Parameter(torch.zeros(n_fixed, K))

        # League specific AR1 processes (rho_l,k)
        # Specified as a real number 
        self.raw_rho = nn.Parameter(torch.zeros(L, K))

        # Cholesky factor for class-specific covariance matrices Sigma_k
        # Initializes as an identity matrix
        self.Sigma = nn.Parameter(torch.eye(L).repeat(K, 1, 1)) 
        
        # Random effects tensor
        self.eps = nn.Parameter(torch.randn(B,L,T,K))

    def compute_random_effects(self):
        """ Generate correlated random effects u_blt,k. """
        u = torch.zeros_like(self.eps)
        u[:,:,0,:] = self.eps[:,:,0,:]
        SigmaHalf = torch.linalg.cholesky(self.Sigma) # Extract Cholesky decomposition of Sigma
        rho = torch.tanh(self.raw_rho) # Force rho between -1 and 1
        # Iterate over timesteps for AR(1)
        for t in range(1, self.T):
            u[:, :, t, :] = rho * u[:, :, t-1, :] + torch.einsum("KLL,BLK->BLK",SigmaHalf,self.eps[:,:,t,:])
        return u
    
    def forward(self, X, batter_ids, league_ids, season_ids):
        """
        X: (n, n_fixed) input fixed effects matrix
        batter_ids: (n,) indices for batter
        league_ids: (n,) indices for league
        season_ids: (n,) indices for season
        """
        # Compute fixed effects
        fixed_effects = X @ self.beta  # (n, K)

        # Compute random effects
        u_blt_k = self.compute_random_effects()  # (B, L, T, K)

        # Gather appropriate random effects for each observation
        random_effects = u_blt_k[batter_ids, league_ids, season_ids, :]  # (n, K)
        
        # Compute final logits
        logits = fixed_effects + random_effects  # (n, K)

        return logits # Logits (no softmax needed, as CrossEntropyLoss applies it) 

    def train_model(self, train_loader, val_loader, criterion, optimizer, num_epochs):
        training_losses = []
        val_losses = []
    
        for epoch in range(num_epochs):
            
            model.train()
            train_loss = 0
            for inputs, labels in train_loader:
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            training_losses.append(train_loss)

            model.eval()
            with torch.no_grad():
                val_loss = 0
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                val_losses.append(val_loss)
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # save model object
        torch.save(model, 'simple_nn.pth')


        # save training history
        history = {'epoch': [num+1 for num in range(num_epochs)], 
                   'training_loss' : training_losses,
                   'validation_loss': val_losses}
        torch.save(history, 'simple_nn_history.pt')

        print('model saved')
    
    def predict(self, X):
        self.eval()

        with torch.no_grad():
            output = self(X)
            _, predicted = torch.max(output.data, 1)
        
        return predicted.numpy()

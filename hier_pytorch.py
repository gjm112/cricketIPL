# Need to Change the Python directory using
# reticulate::use_python("C:/Users/mstuart1/OneDrive - Loyola University Chicago/Documents/.virtualenvs/r-reticulate/Scripts/python.exe")

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import cleaned_data_regression as data

def loss_with_eps_penalty(logits, targets, eps):
    # Cross-entropy loss
    ce_loss = nn.CrossEntropyLoss()(logits, targets)
    
    # Epsilon penalty (prior): sum of squares
    eps_penalty = torch.sum(eps ** 2) / 2
    
    # # Prior: beta ~ N(0, 10^2)
    # beta_prior = torch.sum(beta ** 2) / (2 * 10**2)
    # 
    # # Prior: raw_rho ~ N(0, 10^2)
    # rho_prior = torch.sum(raw_rho ** 2) / (2 * 10**2)
    # 
    # # LKJ-inspired prior on raw_chol
    # lkj_penalty = 0.0
    # for k in range(raw_chol.shape[0]):  # over K
    #     L_k = torch.tril(raw_chol[k])  # (L, L)
    #     Sigma_k = L_k @ L_k.T  # (L, L)
    #     diag = torch.diag(Sigma_k)
    #     stddev = torch.sqrt(diag + 1e-6)
    #     corr = Sigma_k / (stddev[:, None] * stddev[None, :])  # (L, L)
    #     # Penalize off-diagonal terms for deviating from identity
    #     off_diag = corr - torch.eye(corr.shape[0], device=corr.device)
    #     lkj_penalty += torch.sum(off_diag ** 2)
        
    # total_loss = ce_loss + eps_penalty + beta_prior + rho_prior + lkj_penalty
    total_loss = ce_loss + eps_penalty
    return total_loss

class HierarchicalMultinomialRegression(nn.Module):
    def __init__(self, n_fixed, B, L, T, K):
        super().__init__()
        self.n_fixed = n_fixed # Number of fixed effects
        self.B = B # Number of batters
        self.L = L # Number of leagues
        self.T = T # Number of seasons
        self.K = K # Number of possible runs

        # Fixed effects coefficients (Beta)
        self.beta = nn.Parameter(torch.randn(n_fixed, K - 1))

        # League specific AR1 processes (rho_l,k)
        # Specified as a real number 
        self.raw_rho = nn.Parameter(torch.randn(L, K - 1))

        # Cholesky factor for class-specific covariance matrices Sigma_k
        # Initializes as an identity matrix
        self.raw_chol = nn.Parameter(torch.tril(torch.randn(K - 1, L, L)))  # shape: (K - 1, L, L)
        
        # Random effects error tensor
        self.eps = nn.Parameter(torch.randn(B,L,T,K - 1))

    def compute_random_effects(self):
        """ Generate correlated random effects u_blt,k. """
        u_list = []
        # Ensure lower-triangular and make gradients stable
        rho = torch.tanh(self.raw_rho)  # (L, K - 1)

        # First time step
        eps_t = self.eps[:, :, 0, :]  # (B, L, K - 1)
        
        # Ensure rho^2 < 1 for all entries (should already be true)
        rho_squared = rho ** 2  # (L, K - 1)
        # Scaling factor to ensure stationarity
        scaling = 1.0 / torch.sqrt(1.0 - rho_squared)  # (L, K - 1), element-wise
        noise = torch.einsum("KLL,BLK->BLK", self.raw_chol, eps_t)  # (B, L, K - 1)
        u_curr = scaling * noise  # (B, L, K - 1)
        u_list.append(u_curr)
        u_prev = u_curr

        # Iterate over timesteps for AR(1)
        for t in range(1, self.T):
            eps_t = self.eps[:, :, t, :]  # (B, L, K - 1)
            noise = torch.einsum("kij,blk->blj", self.raw_chol, eps_t)  # (B, L, K - 1)
            u_curr = rho * u_prev + noise  # broadcast (L, K - 1) to (B, L, K - 1)
            u_list.append(u_curr)
            u_prev = u_curr

        # Stack along time dimension
        u = torch.stack(u_list, dim=2)  # (B, L, T, K)
        return u
    
    def forward(self, X, batter_ids, league_ids, season_ids):
        """
        X: (n, n_fixed) input fixed effects matrix
        batter_ids: (n,) indices for batter
        league_ids: (n,) indices for league
        season_ids: (n,) indices for season
        """
        # Compute fixed effects
        fixed_effects = X @ self.beta  # (n, K - 1)

        # Compute random effects
        u_blt_k = self.compute_random_effects()  # (B, L, T, K - 1)

        # Gather appropriate random effects for each observation
        random_effects = u_blt_k[batter_ids, league_ids, season_ids, :]  # (n, K - 1)
        
        # Compute final logits
        logits_partial = fixed_effects + random_effects  # (n, K - 1)
        zeros = torch.zeros(logits_partial.size(0), 1, device=logits_partial.device)
        logits = torch.cat([zeros, logits_partial], dim=1)  # (n, K)
        # Extract epsilons for loss function
        eps_sample = self.eps[batter_ids, league_ids, season_ids, :]

        return logits, eps_sample # Logits (no softmax needed, as CrossEntropyLoss applies it) 
    
    def train_model(self, train_loader, val_loader, criterion, optimizer, num_epochs):
        training_losses = []
        val_losses = []
    
        for epoch in range(num_epochs):
            
            model.train()
            train_loss = 0
            for X, batter_ids, league_ids, season_ids, labels in train_loader:
                outputs, eps = self(X, batter_ids, league_ids, season_ids)
                loss = criterion(outputs, labels, eps)

                optimizer.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    loss.backward()
                optimizer.step()

                train_loss += loss.item()
            training_losses.append(train_loss)

            model.eval()
            with torch.no_grad():
                val_loss = 0
                for X, batter_ids, league_ids, season_ids, labels in val_loader:
                    outputs, eps = model(X, batter_ids, league_ids, season_ids)
                    loss = criterion(outputs, labels, eps)
                    val_loss += loss.item()
                val_losses.append(val_loss)
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # save model object
        torch.save(model, 'simple_hier.pth')


        # save training history
        history = {'epoch': [num+1 for num in range(num_epochs)], 
                   'training_loss' : training_losses,
                   'validation_loss': val_losses}
        torch.save(history, 'simple_hier_history.pt')

        print('model saved')
    
    def predict(self, X, batter_ids, league_ids, season_ids):
        self.eval()

        with torch.no_grad():
            output, _ = self(X, batter_ids, league_ids, season_ids)
            _, predicted = torch.max(output, 1)

        return predicted.cpu().numpy()

# load data
batch_size = 128
train_loader = data.create_dataloader(data.train_dataset, batch_size = batch_size, shuffle = True)
val_loader = data.create_dataloader(data.validation_dataset, batch_size = batch_size, shuffle = True)
# X_batch, bat_batch, lg_batch, T_batch, y_batch = next(iter(train_loader))

# Model parameters
n_fixed = data.X.shape[1] # Number of Fixed Effects
B = torch.max(data.bat).to(torch.int) + 1 # Number of batters
L = torch.max(data.lg).to(torch.int) + 1 # Number of leagues
T = torch.max(data.sea).to(torch.int) + 1 # Number of seasons

K = data.y.shape[1]  # Number of output classes

# Initialize model, loss function, and optimizer
model = HierarchicalMultinomialRegression(n_fixed, B, L, T, K)
criterion = loss_with_eps_penalty
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 30

# Train model
model.train_model(train_loader, val_loader, criterion, optimizer, num_epochs)

torch.save(model.state_dict(), 'hier_pytorch.pth')
torch.save(optimizer.state_dict(), 'hier_pytorch_optim.pth')


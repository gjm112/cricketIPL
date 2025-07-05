# Need to Change the Python directory using
# reticulate::use_python("C:/Users/mstuart1/OneDrive - Loyola University Chicago/Documents/.virtualenvs/r-reticulate/Scripts/python.exe")

import torch
import cleaned_data_regression as data # get cleaned data
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset  
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import scale

# import model file for access to model class and training meta (input_size, hidden_size, num_classes, batchsize)
# Be sure to comment out the model training lines first
from hier_pytorch import HierarchicalMultinomialRegression, n_fixed, B, L, T, K, batch_size 

## model evals
model_history = torch.load('simple_hier_history_V2.pt')
trained_model = HierarchicalMultinomialRegression(n_fixed, B, L, T, K)
trained_model.load_state_dict(torch.load('hier_pytorch_V2.pth'))


# training loss progression
lossplot_dat = {'epochs': model_history['epoch'],
                 'training_loss': model_history['training_loss'],
                   'validation_loss' : model_history['validation_loss']}
lossplot_dat = pd.DataFrame(lossplot_dat)

plt.plot(lossplot_dat['epochs'], lossplot_dat['training_loss'], label = 'training loss')
plt.xlabel('Epoch')
plt.title('Training Loss')
plt.show()
plt.plot(lossplot_dat['epochs'], lossplot_dat['validation_loss'], label = 'validation_loss')
plt.xlabel('Epoch')
plt.title('Validation Loss')
plt.show()

## Extract correlation matrix
raw_chol = dict(trained_model.named_parameters())['raw_chol'].data
torch.tril(raw_chol)

cov_tensor = raw_chol @ raw_chol.transpose(-1, -2)  # Ensure positive semi-definite

# Extract standard deviations
stddevs = torch.sqrt(torch.diagonal(cov_tensor, dim1=1, dim2=2))  # shape (5, 5)

# Outer product of stddevs to form denominator for correlation
denom = stddevs.unsqueeze(2) * stddevs.unsqueeze(1)  # shape (5, 5, 5)

# Compute correlation matrices
corr_tensor = cov_tensor / denom

## get model predictions on test set

# get test set
test_loader = data.create_dataloader(data.test_dataset, batch_size = batch_size, shuffle = False)

## initialize
test_predictions = []
test_labels = []

# use model on test set and get predictions
with torch.no_grad():
    for X, batter_id, league_id, season_id in test_loader:
        test_predictions.extend(trained_model.predict(X, batter_id, league_id, season_id))
        _, pred = torch.max(labels, 1)
        test_labels.extend(pred.numpy())

# with torch.no_grad():
#     for X, y, batter_id, league_id, season_id in test_loader:
#         X = X.to(device)
#         y = y.to(device)
#         batter_id = batter_id.to(device)
#         league_id = league_id.to(device)
#         season_id = season_id.to(device)
# 
#         outputs, _ = trained_model(X, batter_id, league_id, season_id)
#         _, predicted = torch.max(outputs, dim=1)
# 
#         all_preds.extend(predicted.cpu().numpy())
#         all_labels.extend(y.cpu().numpy())


conf_matrix = confusion_matrix(test_labels, test_predictions)
accuracy = accuracy_score(test_labels, test_predictions)
report = classification_report(test_labels, test_predictions)

print(f"Accuracy: {accuracy:.4f}")
confmatplot = ConfusionMatrixDisplay(conf_matrix)
confmatplot.plot()
plt.show()
print("Classification Report:\n", report)


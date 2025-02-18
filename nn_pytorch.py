import pandas as pd
import numpy as np
import math
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset  
import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

## read data
bbb = pd.read_csv("data/cricket_data.csv")

select_bbb = pd.DataFrame({'season' : bbb['year'],
  'innings': bbb['innings'],
  'target' : bbb['target'],
  'balls_remaining' : bbb['balls_remaining'],
  'runs_scored_yet' : bbb['runs_scored_yet'],
  'wickets_lost_yet' : bbb['wickets_lost_yet'],
  'venue' : bbb['venue'],
  'striker' : bbb['striker'],
  'bowler': bbb['bowler'],
  'league' : bbb['league']})
 
# clean and normalize
select_bbb = pd.get_dummies(select_bbb, columns = ['season','innings','striker','venue','bowler','league'])
numeric_cols = ['target', 'balls_remaining', 'runs_scored_yet', 'wickets_lost_yet']
select_bbb[numeric_cols] = scale(select_bbb[numeric_cols])

X = torch.tensor(select_bbb.to_numpy(dtype = 'float32'))


bbb_result = pd.DataFrame({'result' : bbb['runs_off_bat']})
bbb_result = pd.get_dummies(bbb_result,columns = ['result'])

y = torch.tensor(bbb_result.values, dtype=torch.float32)


# Create DataLoader
dataset = TensorDataset(X, y)
# Split sizes (e.g., 80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Randomly split dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

## model
# Define a simple neural network
class CategoricalNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CategoricalNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # Logits (no softmax needed, as CrossEntropyLoss applies it)

# Model parameters
input_size = X.shape[1]  # Number of categories in input
hidden_size = int((X.shape[1]+y.shape[1])/2)
num_classes = y.shape[1]  # Number of output classes

# Initialize model, loss function, and optimizer
model = CategoricalNN(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#initialize training stats
training_loss = []


# training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0 # summed loss over all batches in an epoch
    for batch_X, batch_y in train_loader:

      # Forward pass
      outputs = model(batch_X)
      loss = criterion(outputs, batch_y)
  
      # Backward pass and optimization
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
    training_loss.append(total_loss)
    
  
# Save the model output
torch.save(model,"model.pth")

#----------------------

## model evals

model = torch.load('model.pth')

model.eval()

# training loss progression
epochs_list = range(1, 51)
lossplot_dat = {'epochs': epochs_list, 'training_loss': training_loss}
lossplot_dat = pd.DataFrame(lossplot_dat)

plt.plot(lossplot_dat['epochs'], lossplot_dat['training_loss'])



 # initialize model eval stats
all_predictions = []
all_labels = []

# use model on test set and get predictions
with torch.no_grad():  # disable gradient tracking for inference
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)  # forward pass to get batch of logits
        predicted = outputs.argmax(dim=1)
        label = batch_y.argmax(dim=1)
        all_predictions.extend(predicted)
        all_labels.extend(label)



conf_matrix = confusion_matrix(all_labels, all_predictions)
accuracy = accuracy_score(all_labels, all_predictions)
report = classification_report(all_labels, all_predictions)

print(f"Accuracy: {accuracy:.4f}")
confmatplot = ConfusionMatrixDisplay(conf_matrix)
confmatplot.plot()
plt.show()
print("Classification Report:\n", report)

## look into loss function for ordered categories (ex. model pred of 2 is closer to true 3 than true 4)
  # would have to be custom, not sure how doable that actually is
  # could also use regression outputs and mse

## this model includes all possible scoring outcomes, we want to exclude 5s

## would probably be ideal to consider all three train/validation/test sets
  # where validation is used to edit the model and test set is used for comparison across models

import pandas as pd
import numpy as np
import math
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset  

import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

bbb = pd.read_csv("data/cricket_data.csv")

select_bbb = pd.DataFrame({'season' : bbb['year'],
'innings': bbb['innings'],
'target' : bbb['target'],
'balls_remaning' : bbb['balls_remaining'],
'runs_scored_yet' : bbb['runs_scored_yet'],
'wickets_lost_yet' : bbb['wickets_lost_yet'],
'venue' : bbb['venue'],
'striker' : bbb['striker'],
'bowler': bbb['bowler'],
'league' : bbb['league']})

select_bbb = pd.get_dummies(select_bbb, columns = ['season','innings','striker','venue','bowler','league'])
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

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model output
torch.save(outputs,"model.pth")

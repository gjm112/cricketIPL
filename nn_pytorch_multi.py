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

## model
# Define a simple neural network
class CategoricalNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        super(CategoricalNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x  # Logits (no softmax needed, as CrossEntropyLoss applies it)
    
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
        

        
    
# load data
batch_size = 128
train_loader = data.create_dataloader(data.train_dataset, batch_size = batch_size, shuffle = True)
val_loader = data.create_dataloader(data.validation_dataset, batch_size = batch_size, shuffle = True)
data_batch, labels_batch = next(iter(train_loader))


# Model parameters
input_size = data_batch.shape[1]  # Number of categories in input
hidden_size_1 = int(2*(data_batch.shape[1]+labels_batch.shape[1])/3)
hidden_size_2 = int((data_batch.shape[1]+labels_batch.shape[1])/3)
num_classes = labels_batch.shape[1]  # Number of output classes

# Initialize model, loss function, and optimizer
model = CategoricalNN(input_size, hidden_size_1, hidden_size_2, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 30

#train model
model.train_model(train_loader, val_loader, criterion, optimizer, num_epochs)

torch.save(model.state_dict(), 'nn_pytorch_multi.pth')


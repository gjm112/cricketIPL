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
train_loader = data.create_dataloader(data.train_dataset, batch_size = batch_size)
val_loader = data.create_dataloader(data.validation_dataset, batch_size = batch_size)
data_batch, labels_batch = next(iter(train_loader))


# Model parameters
input_size = data_batch.shape[1]  # Number of categories in input
hidden_size = int((data_batch.shape[1]+labels_batch.shape[1])/2)
num_classes = labels_batch.shape[1]  # Number of output classes


# Initialize model, loss function, and optimizer
model = CategoricalNN(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 30

#train model
model.train_model(train_loader, val_loader, criterion, optimizer, num_epochs)



#----------------------

## model evals

model_history = torch.load('simple_nn_history.pt')
trained_model = torch.load('simple_nn.pth')


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

## get model predictions on test set

# get test set
test_loader = data.create_dataloader(data.test_dataset, batch_size = batch_size)

## initialize
test_predictions = []
test_labels = []

# use model on test set and get predictions
with torch.no_grad():
  for inputs, labels in test_loader:
    test_predictions.extend(trained_model.predict(inputs))
    _, pred = torch.max(labels, 1)
    test_labels.extend(pred.numpy())


conf_matrix = confusion_matrix(test_labels, test_predictions)
accuracy = accuracy_score(test_labels, test_predictions)
report = classification_report(test_labels, test_predictions)

print(f"Accuracy: {accuracy:.4f}")
confmatplot = ConfusionMatrixDisplay(conf_matrix)
confmatplot.plot()
plt.show()
print("Classification Report:\n", report)


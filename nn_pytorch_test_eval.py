import torch
import cleaned_data as data # get cleaned data
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset  
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import scale

# import model file for access to model class and training meta (input_size, hidden_size, num_classes, batchsize)
from nn_pytorch import CategoricalNN, input_size, hidden_size, num_classes, batch_size 

## model evals
batch_size = 128
model_history = torch.load('simple_nn_history.pt')
trained_model = CategoricalNN(input_size, hidden_size, num_classes)
trained_model.load_state_dict(torch.load('nn_pytorch.pth'))


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


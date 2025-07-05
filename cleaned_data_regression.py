# Need to Change the Python directory using
# reticulate::use_python("C:/Users/mstuart1/OneDrive - Loyola University Chicago/Documents/.virtualenvs/r-reticulate/Scripts/python.exe")

import pandas as pd
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset  
from sklearn.preprocessing import scale

random.seed(1112)

## read data
bbb = pd.read_csv("data/cricket_data.csv")

## remove non-continuous years (2007 and 2009)
bbb = bbb[bbb['year'] >= 2011]

## remove runs worth 3 and 5
bbb = bbb[bbb['runs_off_bat'].isin([0, 1, 2, 4, 6])]

select_bbb = pd.DataFrame({'innings': bbb['innings'],
  'target' : bbb['target'],
  'balls_remaining' : bbb['balls_remaining'],
  'runs_scored_yet' : bbb['runs_scored_yet'],
  'wickets_lost_yet' : bbb['wickets_lost_yet'],
  'venue' : bbb['venue']})
  
def cat_to_num(vals):
  unique_vals = sorted(vals.unique())
  ret = {vals: i for i, vals in enumerate(unique_vals)}
  return vals.map(ret)

def last_two_digits(vals):
  return str(vals)[-2:]

select_rand = pd.DataFrame({'striker' : bbb['striker'],
  'season' : bbb['season'].apply(last_two_digits),
  'league' : bbb['league']})

# one hot encode and normalize
select_bbb = pd.get_dummies(select_bbb, columns = ['innings','venue'])
select_rand['striker_num'] = cat_to_num(select_rand['striker'])
select_rand['season_num'] = cat_to_num(select_rand['season'])
select_rand['league_num'] = cat_to_num(select_rand['league'])
numeric_cols = ['target', 'balls_remaining', 'runs_scored_yet', 'wickets_lost_yet']
select_bbb[numeric_cols] = scale(select_bbb[numeric_cols])

X = torch.tensor(select_bbb.to_numpy(dtype = 'float32'))
bat = torch.tensor(select_rand['striker_num'].to_numpy(dtype = 'int'))
sea = torch.tensor(select_rand['season_num'].to_numpy(dtype = 'int'))
lg  = torch.tensor(select_rand['league_num'].to_numpy(dtype = 'int'))

bbb_result = pd.DataFrame({'result' : bbb['runs_off_bat']})
bbb_result = pd.get_dummies(bbb_result, columns = ['result'])

y = torch.tensor(bbb_result.values, dtype=torch.float32)


# Create Dataset
dataset = TensorDataset(X, bat, lg, sea, y)

# Split sizes (e.g., 75% train, 10% validation, 15% test)
train_size = int(0.75 * len(dataset))
validation_size = int(0.1*len(dataset))
test_size = len(dataset) - train_size - validation_size


# Randomly split dataset
train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])


# turn subsets back into datasets

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
        

train_dataset = CustomDataset(train_dataset)
validation_dataset = CustomDataset(validation_dataset)
test_dataset = CustomDataset(test_dataset)



def create_dataloader(data, batch_size, shuffle):
    dataset = CustomDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = shuffle)
    return dataloader





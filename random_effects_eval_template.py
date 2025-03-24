# template model evaluations
## need model class and parameters along with trained model  

## libraries

import torch
import cleaned_data as data # get cleaned data
import pandas as pd
from sklearn.preprocessing import scale
from torch.utils.data import DataLoader, TensorDataset, Dataset  

# import model file for access to model class and training meta (input_size, hidden_size, num_classes, batchsize)
from nn_pytorch import CategoricalNN, input_size, hidden_size, num_classes, batch_size 


# take in model
model = CategoricalNN(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('nn_pytorch.pth'))

# get bbb (cleaned but has all original columns and pre-dummy variables)
bbb = data.bbb

## create data for each batter, for each season they play over all pitches in each league

# get list of batters per season
batters_per_season = bbb.groupby(['year'])['striker'].value_counts().reset_index()
    # number of unique batters per season : batters_per_season.groupby(['year'])['year'].value_counts().reset_index())

# get list of seasons present in data
list_of_seasons = bbb['year'].unique().tolist()

def replace_value(value):
    value_map = {
        3: 4,
        4: 6
    }
    return value_map.get(value, value)

# save outputs into nested dict where each player has a key, and a separate (key, value) for each season they played in
player_season_dict ={}

# for each batter in each season they appear, replicate all pitches in that season with them as the batter
for year in list_of_seasons[0:2]:
    print(year)
    list_of_batters = batters_per_season[batters_per_season['year'] == year]['striker'].to_list()
    season_col = f'season_{year}'

    for batter in list_of_batters[0:5]:
        print(batter)
        tensor_shape_bbb = data.select_bbb[data.select_bbb[season_col] == 1]

        striker_cols = [column for column in tensor_shape_bbb if column.startswith('striker_')]
        tensor_shape_bbb[striker_cols] = 0

        current_striker = f'striker_{batter}'
        tensor_shape_bbb[current_striker] = 1


        # turn into tensordataset
        X = torch.tensor(tensor_shape_bbb.to_numpy(dtype = 'float32'))
        X = TensorDataset(X)

        
        player_season_loader = data.create_dataloader(X, batch_size =  batch_size)

        player_season_preds = []
        with torch.no_grad():
            for input in player_season_loader:

                preds = model.predict(input[0])
                #preds = [int(pred) for pred in preds]
                preds = [replace_value(int(pred)) for pred in preds]

                player_season_preds.extend(preds)
        
        
        # save player_season preds as larger data object
        if batter not in player_season_dict:
            player_season_dict[batter] = {}

        player_season_dict[batter][year] = player_season_preds

                                                      

for batter, season in player_season_dict.items():
    print(batter)
    for year, preds in season.items():
        print(year)
        print(preds)




## average over batter/league/season






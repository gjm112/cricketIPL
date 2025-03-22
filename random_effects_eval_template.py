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


# for each batter in each season they appear, replicate all pitches in that season with them as the batter
for year in list_of_seasons:

    list_of_batters = batters_per_season[batters_per_season['year'] == year]['striker'].to_list()

    for batter in list_of_batters:

        modified_bbb = bbb[bbb['year'] == year].copy(deep = True) # select balls only from current season
        modified_bbb['striker'] = str(batter) # insert current batter as batter over all pitches

        # put into model form (only selected columns, onehot categorical vars)
        tensor_shape_bbb = pd.DataFrame({'season' : modified_bbb['year'],
            'innings': modified_bbb['innings'],
            'target' : modified_bbb['target'],
            'balls_remaining' :modified_bbb['balls_remaining'],
            'runs_scored_yet' : modified_bbb['runs_scored_yet'],
            'wickets_lost_yet' : modified_bbb['wickets_lost_yet'],
            'venue' :modified_bbb['venue'],
            'striker' : modified_bbb['striker'],
            'bowler': modified_bbb['bowler'],
            'league' : modified_bbb['league']})
        
        tensor_shape_bbb = pd.get_dummies(tensor_shape_bbb, columns = ['season','innings','striker','venue','bowler','league'])
        numeric_cols = ['target', 'balls_remaining', 'runs_scored_yet', 'wickets_lost_yet']
        tensor_shape_bbb[numeric_cols] = scale(tensor_shape_bbb[numeric_cols])
        

        # turn into tensordataset
        X = torch.tensor(tensor_shape_bbb.to_numpy(dtype = 'float32'))
        X = TensorDataset(X)

        player_season_loader = data.create_dataloader(X, batch_size =  batch_size)

        player_season_preds = []
        with torch.no_grad():
            for input in player_season_loader:
                player_season_preds.extend(model.predict(input))

        # save player_season preds as larger data object
            ## START HERE
                # need to save (player, season) as key and predictions as entry in dict-like object


                                                      







## get predicted response

## average over batter/league/season






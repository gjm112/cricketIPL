# template model evaluations
## need model class and parameters along with trained model  

## libraries

import torch
import cleaned_data_regression as data # get cleaned data
import pandas as pd
from sklearn.preprocessing import scale
from torch.utils.data import DataLoader, TensorDataset, Dataset  

# import model file for access to model class and training meta (input_size, hidden_size, num_classes, batchsize)
from hier_pytorch import HierarchicalMultinomialRegression, n_fixed, B, L, T, K, batch_size 


# take in model
model = HierarchicalMultinomialRegression(n_fixed, B, L, T, K)
model.load_state_dict(torch.load('hier_pytorch.pth'))

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
rob_estimates_dfs = []

# for each batter in each season they appear, replicate all pitches in that season with them as the batter
for year in list_of_seasons:
    print(year)

    # get all batters who played in the current season
    list_of_batters = batters_per_season[batters_per_season['year'] == year]['striker'].to_list()
    season_col = f'season_{year}'

    season_bbb = bbb[bbb['year'] == year].copy()

    for batter in list_of_batters:
        print(batter)
        print(f' {list_of_batters.index(batter)} / {len(list_of_batters)}')

        # select only pitches thrown in the current season
        tensor_shape_bbb = data.select_bbb[data.select_bbb[season_col] == 1] 

        # set striker indicator col to current batter
        striker_cols = [column for column in tensor_shape_bbb if column.startswith('striker_')]
        tensor_shape_bbb.loc[:,striker_cols] = bool(0)

        current_striker = f'striker_{batter}'
        tensor_shape_bbb.loc[:,current_striker] = bool(1)

        # turn into tensordataset
        X = torch.tensor(tensor_shape_bbb.to_numpy(dtype = 'float32'))
        X = TensorDataset(X)

        # load for prediction with shuffle = False to maintain index order
        player_season_loader = data.create_dataloader(X, batch_size =  batch_size, shuffle = False)

        # predict for current player over all balls (in all leagues) thrown in the current season
        player_season_preds = []
        with torch.no_grad():
            for input in player_season_loader:

                preds = model.predict(input[0]) # predict
                preds = [replace_value(int(pred)) for pred in preds] # insert true run values

                player_season_preds.extend(preds)
        
        
        # save player preds for each season as nested dictionary with {batter : (season: preds)}
        if batter not in player_season_dict:
            player_season_dict[batter] = {}

        player_season_dict[batter][year] = player_season_preds

        # connect predictions back to plays (shuffle = False in dataloader should preserve index order)
        season_bbb.loc[:,'model_pred'] = player_season_preds

        # find average within each league
        pred_rob = season_bbb.groupby('league')['model_pred'].mean().reset_index()
        true_rob = season_bbb[season_bbb['striker'] == batter].groupby('league')['runs_off_bat'].mean().reset_index()
        
        # join preds and true with NA for true where player did not play in league
        joined = pd.merge(pred_rob, true_rob, on = 'league', how='outer')

        joined['year']= year
        joined['striker'] = batter

        rob_estimates_dfs.append(joined)


rob_estimates = pd.concat(rob_estimates_dfs, ignore_index = True)
rob_estimates.to_csv('nn_pytorch_ranefs.csv')
        


                                 
# to iterate through nested dict and get predictions:
""" for batter, season in player_season_dict.items():
    print(batter)
    for year, preds in season.items():
        print(year)
        print(preds) """




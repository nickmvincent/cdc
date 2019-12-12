#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os

#%%
cdc_seed = 100

#%%
kf = KFold(n_splits=10, shuffle=True, random_state=0)

data_folder = './data'
preds_folder = './preds'
dataset_folder = 'breast_cancer'

if not os.path.isdir(f'{preds_folder}/{dataset_folder}'):
    os.mkdir(f'{preds_folder}/{dataset_folder}')


pre = f'{data_folder}/{dataset_folder}'
dataset_file = 'sklearn_breast_cancer.csv'
USER_SPLIT = False

#%%
df = pd.read_csv(f'{pre}/{dataset_file}')
df.head()
# for ml-1m
#df = pd.read_csv(f'{pre}/{dataset_file}', header=0, names=['user', 'item', 'rating', 'timestamp'], sep="::")

#%%
hidden_test_df = df.sample(frac=0.25, random_state=0)
hidden_test_df.to_csv(f'{data_folder}/{dataset_folder}/hidden_test.csv')
train_df = df.drop(hidden_test_df.index)

#%%
if USER_SPLIT:
    users = pd.Series(train_df['user'].unique())
    print(users.head(3))

#%%
fracs = []
for i in range(1, 20):
    fracs.append(round(i * 0.05, 2))
fracs

#%%
seeds = [0,1,2,3,4]
for frac in fracs:
    for seed in seeds:
        scenario = f'{frac}_random{seed}'
        print(f'scenario: {scenario}')

        if USER_SPLIT:
            strikers = users.sample(frac=frac, random_state=seed)
            print('# strikers', len(strikers))

            mask = df['user'].isin(strikers)
            print('# ratings for strikers', sum(mask))
        else:
            np.random.seed(seed)
            small_mask = np.random.rand(len(train_df)) < frac

        df_large = train_df[~small_mask]
        df_small = train_df[small_mask]

        subdir = f'{pre}/{scenario}'
        if not os.path.isdir(subdir):
            os.mkdir(subdir)
        preds_dir = f'{preds_folder}/{dataset_folder}'

        if not os.path.isdir(preds_dir):
            os.mkdir(preds_dir)

        for (subdf, name) in (
            (df_large, 'large'),
            (df_small, 'small'),
        ):
            for i, (train_index, test_index) in enumerate(kf.split(subdf)):
                subdf.iloc[train_index].to_csv(f'{subdir}/{name}_train{i}.csv', header=None, index=None)
                subdf.iloc[test_index].to_csv(f'{subdir}/{name}_test{i}.csv', header=None, index=None)

#tr -d \' < train0.csv > train0.csv
#tr -d \' < test0.csv > test0.csv

#%%

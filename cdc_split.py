#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

import os
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
if USER_SPLIT:
    users = pd.Series(df['user'].unique())
    print(users.head(3))

#%%
for frac in [
        # 0.01,
        # 0.02,
        # 0.03,
        # 0.04,
        # 0.05, 
        # .10,
        # .15,
        # .2
        0.4
    ]:
    scenario = f'{frac}_random'
    print(f'scenario: {scenario}')

    if USER_SPLIT:
        strikers = users.sample(frac=frac, random_state=100)
        print('# strikers', len(strikers))

        mask = df['user'].isin(strikers)
        print('# ratings for strikers', sum(mask))
    else:
        np.random.seed(100)
        mask = np.random.rand(len(df)) < frac

    df_large = df[~mask]
    df_small = df[mask]

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
            print(train_index)
            subdf.iloc[train_index].to_csv(f'{subdir}/{name}_train{i}.csv', header=None, index=None)
            subdf.iloc[test_index].to_csv(f'{subdir}/{name}_test{i}.csv', header=None, index=None)
            break

#tr -d \' < train0.csv > train0.csv
#tr -d \' < test0.csv > test0.csv

#%%

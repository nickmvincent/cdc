#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os


#%%
# fracs = []
# for i in range(1, 20):
#     fracs.append(round(i * 0.05, 2)
fracs = [0.01, 0.05, .1, .2, .3, .4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
seeds = [0,1,2,3,4]

fracs



#%%
dataset = 'pinterest-20'


kf = KFold(n_splits=10, shuffle=True, random_state=0)
data_folders = {
    'pinterest-20': '/Users/nick/workspaces/RecSys2019/Conferences/WWW/NeuMF_github/Data',
    'breast_cancer': './data/breast_cancer',
    'ml-10m': './data/breast_cancer'
}
data_folder = data_folders[dataset]
preds_folder = './preds'

if not os.path.isdir(f'{preds_folder}/{dataset}'):
    os.mkdir(f'{preds_folder}/{dataset}')


USER_SPLIT = False

EVAL = 'kfold'
# other options include other loo (leave-one-out), other kfold, ...

if dataset == 'breast_cancer':
    dataset_file = 'sklearn_breast_cancer.csv'
    df = pd.read_csv(f'{data_folder}/{dataset_file}')
elif dataset =='ml-1m':
    df = pd.read_csv(f'{data_folder}/{dataset_file}', header=None, names=['user', 'item', 'rating', 'timestamp'], sep="::")
    USER_SPLIT = True
elif dataset =='pinterest-20':
    train_file = 'pinterest-20.train.rating'
    test_file = 'pinterest-20.test.rating'
    neg_file = 'pinterest-20.test.negative'
    df = pd.read_csv(f'{data_folder}/{train_file}', header=None, names=['user', 'item', 'rating', 'timestamp'], sep="\t")
    test_df = pd.read_csv(f'{data_folder}/{test_file}', header=None, names=['user', 'item', 'rating', 'timestamp'], sep="\t")
    neg_df = pd.read_csv(f'{data_folder}/{neg_file}', header=None, sep="\t")
    USER_SPLIT = True
    EVAL = 'loo'
df.head(3)


#%%
if EVAL == 'kfold':
    hidden_test_df = df.sample(frac=0.25, random_state=0)
    hidden_test_df.to_csv(f'{data_folder}/{dataset}/hidden_test.csv')
    train_df = df.drop(hidden_test_df.index)
else:
    train_df = df

#%%
if USER_SPLIT:
    users = pd.Series(train_df['user'].unique())
    print(users.head(3))

#%%
for frac in fracs:
    for seed in seeds:
        scenario = f'{frac}_random{seed}'
        print(f'scenario: {scenario}')

        if USER_SPLIT:
            strikers = users.sample(frac=frac, random_state=seed)
            print('# strikers', len(strikers))

            small_mask = df['user'].isin(strikers)
            print('# ratings for strikers', sum(small_mask))
        else:
            np.random.seed(seed)
            small_mask = np.random.rand(len(train_df)) < frac

        
        preds_dir = f'{preds_folder}/{dataset}'

        if not os.path.isdir(preds_dir):
            os.mkdir(preds_dir)

        if EVAL == 'kfold':
            subdir = f'{data_folder}/{scenario}'
            if not os.path.isdir(subdir):
                os.mkdir(subdir)
            df_large = train_df[~small_mask]
            df_small = train_df[small_mask]
            for (subdf, name) in (
                (df_large, 'large'),
                (df_small, 'small'),
            ):
                for i, (train_index, test_index) in enumerate(kf.split(subdf)):
                    subdf.iloc[train_index].to_csv(f'{subdir}/{name}_train{i}.csv', header=None, index=None)
                    subdf.iloc[test_index].to_csv(f'{subdir}/{name}_test{i}.csv', header=None, index=None)
        elif EVAL == 'loo':
            # TODO probably need to reset item indices as well.
            # should we just use reindex??
            
            # ===
            # Make subdirs
            subdir_large = f'{data_folder}/{scenario}_large'
            if not os.path.isdir(subdir_large):
                os.mkdir(subdir_large)

            subdir_small = f'{data_folder}/{scenario}_small'
            if not os.path.isdir(subdir_small):
                os.mkdir(subdir_small)

            # hack to avoid re-running
            try:
                pd.read_csv('{subdir_large}/{dataset}.test.rating')
                continue
            except:
                pass

            # ===
            # Test Data
            test_small_mask = test_df['user'].isin(strikers)

            test_large = test_df[~test_small_mask]
            test_small = test_df[test_small_mask]

            old_to_new_large = {}
            for i, (uid, row) in enumerate(test_large.iterrows()):
                old_to_new_large[row.user] = i
            test_large.loc[:,'user'] = test_large['user'].map(old_to_new_large)

            old_to_new_small = {}
            for i, (uid, row) in enumerate(test_small.iterrows()):
                old_to_new_small[row.user] = i
            test_small.loc[:, 'user'] = test_small['user'].map(old_to_new_small)

            test_large.to_csv(f'{subdir_large}/{dataset}.test.rating', header=None, index=None, sep='\t')
            test_small.to_csv(f'{subdir_small}/{dataset}.test.rating', header=None, index=None, sep='\t')

            # ===
            # Train Data

            train_large = train_df[~small_mask]
            train_large.loc[:, 'user'] = train_large['user'].map(old_to_new_large)
            train_small = train_df[small_mask]
            train_small.loc[:, 'user'] = train_small['user'].map(old_to_new_small)

            train_large.to_csv(f'{subdir_large}/{dataset}.train.rating', header=None, index=None, sep='\t')
            train_small.to_csv(f'{subdir_small}/{dataset}.train.rating', header=None, index=None, sep='\t')

            # ===
            # Negative Data

            neg_large = neg_df[~test_small_mask]
            neg_small = neg_df[test_small_mask]
            neg_large.to_csv(f'{subdir_large}/{dataset}.test.negative', header=None, index=None, sep='\t')
            neg_small.to_csv(f'{subdir_small}/{dataset}.test.negative', header=None, index=None, sep='\t')
#%%

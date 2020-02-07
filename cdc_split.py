#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, save_npz
import sys
# import argparse

# parser = argparse.ArgumentParser(description='Split data for a data leverage campaign')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')

# args = parser.parse_args()
#%%
# fracs = []
# for i in range(1, 20):
#     fracs.append(round(i * 0.05, 2)
#if fracs == 'full':
fracs = [0.01, 0.05, .1, .2, .3, .4, 0.5,]
seeds = [0,1,2,3,4]
seeds = [5,6,7,8,9]
fracs

#%%
dataset = sys.argv[1]


kf = KFold(n_splits=10, shuffle=True, random_state=0)
data_folders = {
    'pinterest-20': './RecSys2019/Conferences/WWW/NeuMF_github/Data',
    'breast_cancer': './data/breast_cancer',
    'ml-10m': './libFM/libfm-1.42.src/data/ml-10m',
    'toxic': './data/toxic',
}
data_folder = data_folders[dataset]
preds_folder = './preds'

if not os.path.isdir(f'{preds_folder}/{dataset}'):
    os.mkdir(f'{preds_folder}/{dataset}')

USER_SPLIT = False

EVAL = 'kfold'
# other options include other loo (leave-one-out), other kfold, ...

# @FUTURE: could be abstracted into config files...
if dataset == 'breast_cancer':
    dataset_file = 'sklearn_breast_cancer.csv'
    df = pd.read_csv(f'{data_folder}/{dataset_file}')
elif dataset =='ml-10m':
    dataset_file = 'ratings.dat'
    df = pd.read_csv(f'{data_folder}/{dataset_file}', engine='python', header=None, names=['user', 'item', 'rating', 'timestamp'], sep="::")
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
elif dataset == 'toxic':
    print('Loading toxic')
    dataset_file = 'train.csv'
    df = pd.read_csv(f'{data_folder}/{dataset_file}').fillna(' ')
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['binary_label'] = df[class_names].any(axis=1)
    

df.head(3)


#%%
HIDDEN_FRAC = 0.1
HIDDEN_SEED = 0
if EVAL == 'kfold':
    hidden_test_df = df.sample(frac=HIDDEN_FRAC, random_state=HIDDEN_SEED)
    hidden_test_df.to_csv(f'{data_folder}/hidden_test.csv')
    train_df = df.drop(hidden_test_df.index)
# elif EVAL == 'loo':
#     hidden_test_users = users.sample(frac=0.1, random_state=HIDDEN_SEED)
else:
    train_df = df


#%%
if USER_SPLIT:
    users = pd.Series(df['user'].unique())
    print(users.head(3))

VALID_FRAC = 0.1
VALID_SEED = 100
#%%
for frac in fracs:
    for seed in seeds:
        scenario = f'{frac}_{tactic}{seed}'
        print(f'scenario: {scenario}')

        if USER_SPLIT:
            small_users = users.sample(frac=frac, random_state=seed)
            small_mask = train_df['user'].isin(small_users)

            # if EVAL == 'loo': # we will need these below
            #     large_users = users.drop(small_users.index)

            #     small_valid_users = small_users.sample(frac=VALID_FRAC, random_state=VALID_SEED)
            #     small_valid_mask = users.isin(small_valid_users)

            #     large_valid_users = large_users.sample(frac=VALID_FRAC, random_state=VALID_SEED+1)
            #     large_valid_mask = users.isin(large_valid_users)

            df_large = train_df[~small_mask]
            df_small = train_df[small_mask] 

            if tactic == 'randomvandal':
                df_vandalised = df_small.copy()
                df_vandalised.rating = np.random.randint(1,11, len(df_vandalised)) / 2
                df_large = pd.concat([df_large, df_vandalised])
            elif tactic == 'extremevandal':
                df_vandalised = df_small.copy()
                df_vandalised.loc[df_vandalised.rating < 3] = 1.0
                df_vandalised.loc[df_vandalised.rating >= 3] = 5.0
                df_large = pd.concat([df_large, df_vandalised])
            print('# strikers', len(small_users))
            print('# ratings for strikers', sum(small_mask))

            

        else:
            df_small = train_df.sample(frac=frac,random_state=seed) #random state is a seed value
            df_large = train_df.drop(df_small.index)
        assert len(df_large) + len(df_small) == len(train_df)
        
        preds_dir = f'{preds_folder}/{dataset}'

        if not os.path.isdir(preds_dir):
            os.mkdir(preds_dir)

        # Toxic and ML-10M
        if EVAL == 'kfold':
            subdir = f'{data_folder}/{scenario}'
            if not os.path.isdir(subdir):
                os.mkdir(subdir)

            for (subdf, name) in (
                (df_large, 'large'),
                (df_small, 'small'),
            ):
                # Toxic
                if dataset == 'toxic':
                    text = subdf['comment_text']
                    labels = subdf['binary_label']
                    assert(any(text.isna()) == False)                    

                    test_text = hidden_test_df['comment_text']
                    test_labels = hidden_test_df['binary_label']

                    for i, (train_index, valid_index) in enumerate(kf.split(text)):
                        valid_text = text.iloc[valid_index]
                        valid_labels = labels.iloc[valid_index]
                        #print(train_index)
                        train_text = text.iloc[train_index]
                        train_labels = labels.iloc[train_index]                        
                        assert len(valid_text) + len(train_text) == len(text)
                        assert(any(valid_text.isna()) == False)
                        assert(any(train_text.isna()) == False)                        

                        word_vectorizer = TfidfVectorizer(
                            sublinear_tf=True,
                            strip_accents='unicode',
                            analyzer='word',
                            token_pattern=r'\w{1,}',
                            stop_words='english',
                            ngram_range=(1, 1),
                            max_features=10000)
                        #print(train_text)
                        word_vectorizer.fit(train_text)
                        train_word_features = word_vectorizer.transform(train_text)
                        test_word_features = word_vectorizer.transform(test_text)
                        valid_word_features = word_vectorizer.transform(valid_text)
                        char_vectorizer = TfidfVectorizer(
                            sublinear_tf=True,
                            strip_accents='unicode',
                            analyzer='char',
                            stop_words='english',
                            ngram_range=(2, 6),
                            max_features=50000)
                        char_vectorizer.fit(train_text)
                        train_char_features = char_vectorizer.transform(train_text)
                        test_char_features = char_vectorizer.transform(test_text)
                        valid_char_features = char_vectorizer.transform(valid_text)

                        train_features = hstack([train_char_features, train_word_features])
                        test_features = hstack([test_char_features, test_word_features])
                        valid_features = hstack([valid_char_features, valid_word_features])
                    
                        save_npz(f'{subdir}/{name}_train', train_features)
                        np.savez(f'{subdir}/{name}_train_labels', labels=train_labels.values)

                        save_npz(f'{subdir}/{name}_valid', valid_features)
                        np.savez(f'{subdir}/{name}_valid_labels', labels=valid_labels.values)

                        save_npz(f'{subdir}/{name}_hidden', test_features)
                        np.savez(f'{subdir}/{name}_hidden_labels', labels=test_labels.values)
                        break
                        # to do just 1 fold
                else:
                    for i, (train_index, valid_index) in enumerate(kf.split(subdf)):
                        subdf.iloc[train_index].to_csv(f'{subdir}/{name}_train{i}.csv', header=None, index=None)
                        # this is a bit of a hack b/c LIBFM doesn't save Bayesian MF models.
                        concat_test = pd.concat([
                            hidden_test_df, subdf.iloc[valid_index]
                        ])
                        concat_test.to_csv(f'{subdir}/{name}_test{i}.csv', header=None, index=None)
                        break # to do just 1 fold
        elif EVAL == 'loo':
            # ===
            # Make subdirs
            subdir_large = f'{data_folder}/{scenario}_large'
            if not os.path.isdir(subdir_large):
                os.mkdir(subdir_large)

            subdir_small = f'{data_folder}/{scenario}_small'
            if not os.path.isdir(subdir_small):
                os.mkdir(subdir_small)

            # hack to avoid re-running
            # try:
            #     pd.read_csv('{subdir_large}/{dataset}.test.rating')
            #     continue
            # except:
            #     pass

            # ===
            # Test Data
            test_small_mask = test_df['user'].isin(small_users)
            test_large = test_df[~test_small_mask]
            test_small = test_df[test_small_mask]

            # to re-index the users ids
            # so small starts at 0 and large starts at 0
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

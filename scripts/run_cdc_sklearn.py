#%%
# see also this version from leaderboards: https://www.kaggle.com/guocan/logistic-regression-with-words-and-char-n-g-13417e

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.sparse import load_npz
from sklearn.metrics import roc_auc_score


#%%
data_folder = './data'
preds_folder = './preds'
dataset = 'toxic'

#%%
if dataset == 'toxic':
    pass#hidden = load_npz(f'{data_folder}/{dataset}/hidden_test_processed.npz')
else:
    hidden = pd.read_csv(f'{data_folder}/{dataset}/hidden_test.csv', index_col=0)
    print(hidden.head(3))

fracs = [0.01, 0.05, .1, .2, .3, .4, .5]
fracs

#%%
seeds = [0,1,2,3,4]
seeds = [0]


row_dicts = []
for frac in fracs:
    for seed in seeds:
        scenario = f'{frac}_random{seed}'
        print(scenario)
        for co in ['small', 'large']:
            pre = f'{data_folder}/{dataset}/{scenario}'

            if dataset == 'toxic':
                clf = LogisticRegression(C=0.1, solver='sag', random_state=0)
                train = load_npz(f'{pre}/{co}_train.npz')
                train_labels = np.load(f'{pre}/{co}_train_labels.npz')['labels']

                hidden = load_npz(f'{pre}/{co}_hidden.npz')
                hidden_labels = np.load(f'{pre}/{co}_hidden_labels.npz')['labels']


                clf.fit(train, train_labels)
                probs = clf.predict_proba(hidden)[:, 1]

                roc = roc_auc_score(hidden_labels, probs)
                row_dicts.append({
                    'frac': frac,
                    'company': co,
                    'hidden_score': roc,
                    'len(train_y)': len(train_labels),
                    'seed': seed,
                })
            else:
                for fold in range(0, 10):
                    test = pd.read_csv(f'{pre}/{co}_test{fold}.csv', header=None)
                    train = pd.read_csv(f'{pre}/{co}_train{fold}.csv', header=None)

                    #clf = RandomForestClassifier(max_depth=2, random_state=0)
                    clf = LogisticRegression(random_state=0)

                    train_X = train.iloc[:,:-1].values
                    train_y = train.iloc[:,-1].values

                    test_X = test.iloc[:,:-1].values
                    test_y = test.iloc[:,-1].values

                    hidden_X = hidden.iloc[:,:-1].values
                    hidden_y = hidden.iloc[:,-1].values


                    clf.fit(train_X, train_y)

                    score = clf.score(test_X, test_y)

                    hidden_score = clf.score(hidden_X, hidden_y)

                    row_dicts.append({
                        'frac': frac,
                        'company': co,
                        'score': score,
                        'hidden_score': hidden_score,
                        'len(train_y)': len(train_y),
                        'fold': fold,
                        'seed': seed,
                    })

res = pd.DataFrame(row_dicts)
res.to_csv(f'results/{dataset}_rows.csv')

# %%

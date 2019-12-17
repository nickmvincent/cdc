#%%
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


#%%
data_folder = './data'
preds_folder = './preds'
dataset_folder = 'breast_cancer'

#%%
hidden = pd.read_csv(f'{data_folder}/{dataset_folder}/hidden_test.csv', index_col=0)
hidden.head()

#%%
#%%
fracs = []
for i in range(1, 20):
    fracs.append(round(i * 0.05, 2))
fracs

#%%
seeds = [0,1,2,3,4]
#seeds = [0,1]


row_dicts = []
for frac in fracs:
    for seed in seeds:
        scenario = f'{frac}_random{seed}'
        print(scenario)
        for co in ['small', 'large']:
            pre = f'{data_folder}/{dataset_folder}/{scenario}'
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
res.to_csv(f'{dataset_folder}_rows.csv')

#%%
res
# %%
res[res.co == 'small'].groupby('frac').mean().plot(y='hidden_score')

#%%
#res[(res.co == 'small') & (res.seed == 3)].groupby('frac').mean().plot(y='hidden_score')


#%% 
res[res.co == 'large'].groupby('frac').mean().plot(y='hidden_score')

# %%

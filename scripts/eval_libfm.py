
#%%
from sklearn.metrics import mean_squared_error
import pandas as pd
from math import sqrt
import numpy as np

def rmse_score(y, yhat):
    """
    Root mean squared error
    """
    return sqrt(mean_squared_error(y, yhat))

#%%
pre = 'libFM/libfm-1.42.src'
#%%

seeds = [0,1,2,3,4]
fracs = ['0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5']
rows = []
hidden = pd.read_csv(
    f'{pre}/data/ml-10m/hidden_test.csv',
    header=0, names=['user', 'item', 'rating', 'timestamp'])
for frac in fracs:
    for seed in seeds:
        scenario = f'{frac}_random{seed}'
        for entity in [
            'small', 
            'large'
        ]:
            try:
                preds = pd.read_csv(
                    f'{pre}/preds/ml-10m/{scenario}/{entity}_test0.preds',
                    header=0, names=['rating'])

                truth = pd.read_csv(
                    f'{pre}/data/ml-10m/{scenario}/{entity}_test0.csv',
                    header=0, names=['user', 'item', 'rating', 'timestamp'])

                valid = truth.iloc[:len(hidden)]
                test = truth.iloc[len(hidden):]
                assert test.iloc[0] == hidden.iloc[0]
                assert test.iloc[-1] == hidden.iloc[-1]
                
            except Exception as e:
                print(e)
                print(f'Failed for {scenario}/{entity}')
                continue
            
            row = {
                'frac': frac,
                'scenario': scenario,
                'company': entity,
                'valid_rmse': rmse_score(valid.rating, preds.rating),
                'hidden_rmse': rmse_score(test.rating, preds.rating),
                'seed': seed,
            }
            # for score in [
            #     rmse_score,
            #     #surfaced_hits,
            # ]:
            #     val = score(truth.rating, preds.rating)
            #     if isinstance(val, dict):
            #         for k, v in val.items():
            #             col = score.__name__ + '_' + k
            #             row[col] = v
            #             #row[col + '_mean_baseline'] = mean_baseline
            #     else:
            #         col = score.__name__
            #         row[col] = val
            #         diff_from_itemmean = itemmean[score.__name__] - val
            #         diff_needed_for_sota = itemmean[score.__name__] - sota[score.__name__]
            #         diff_from_sota = sota[score.__name__] - val
            #         fraction_to_sota = diff_from_itemmean / diff_needed_for_sota
            #         row['diff_from_sota'] = diff_from_sota
            #         row['fraction_to_sota'] = fraction_to_sota

            rows.append(row)


#%%
results = pd.DataFrame(rows).dropna()

#%%
results.to_csv('results/ml-10m_rows.csv')

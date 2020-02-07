
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
                config = '1,1,32_50'
                preds = pd.read_csv(
                    f'{pre}/preds/ml-10m/{scenario}/{config}/{entity}_test0.preds',
                    header=None, names=['rating'])

                truth = pd.read_csv(
                    f'{pre}/data/ml-10m/{scenario}/{entity}_test0.csv',
                    header=None, names=['user', 'item', 'rating', 'timestamp'])
                assert len(preds) == len(truth)
                n = len(hidden)
                print(n)
                test_preds = preds.iloc[:n]
                test = truth.iloc[:n]

                valid_preds = preds.iloc[n:]
                valid = truth.iloc[n:]
                #print('hidden\n', hidden.head(3).user, '\ntestn\n', test.head(3).user)
                assert len(test) + len(valid) == len(truth)
                assert hidden.user.iloc[0] == test.user.iloc[0]
                #print(truth.head(3))
                #print(hidden.head(3))
                #print('test')
                #print(test.head(3))
                
            except Exception as e:
                print(e)
                print(f'Failed for {scenario}/{entity}')
                continue
            
            row = {
                'frac': frac,
                'scenario': scenario,
                'company': entity,
                'valid_rmse': rmse_score(valid.rating, valid_preds.rating),
                'hidden_rmse': rmse_score(test.rating, test_preds.rating),
                'seed': seed,
                '# hidden': n,
                '# valid': len(valid),
            }
            print(row)
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
results.to_csv(f'results/ml-10m_{config}rows.csv')

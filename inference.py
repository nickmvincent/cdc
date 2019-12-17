#%%
from scipy.stats import poisson
import pandas as pd
import matplotlib.pyplot as plt

# data here: https://github.com/fivethirtyeight/uber-tlc-foil-response


#%%
def confidence_width(count):
    ci_low, ci_upp = poisson.interval(0.95, count)
    #print(ci_low, ci_upp)
    return ci_upp - ci_low

#%%
row_dicts = []

df = pd.read_csv('uber-raw-data-apr14.csv')
df[['Date', 'Time']] = df['Date/Time'].str.split(' ', expand=True)

#%%
df.head()

#%%

for frac in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    samp = df.sample(frac=frac)
    counts = samp.groupby('Date').Lat.count()
    widths = counts.apply(lambda x: confidence_width(x))
    #print(counts)
    #print(widths)
    val = widths.mean() / frac
    print(frac, val)
    row_dicts.append({
        'frac': frac,
        'val': val,
    })

worst = row_dicts[0]['val']
best = row_dicts[-1]['val']
res = pd.DataFrame(row_dicts)
res.to_csv('inference_rows.csv', index=None)

# %%
res['diff_from_worst'] = worst - res['val']
diff_from_worst_to_best = worst - best

res['copy_ratio'] = res.diff_from_worst / diff_from_worst_to_best
res['competitor'] = 0

for i, row in res.iterrows():
    complement = round(1 - row.frac, 2)
    print(complement)
    # assumes there's only one match
    complement_val = res[res.frac == complement].diff_from_worst.values[0]
    print('cv', complement_val)
    res.loc[i, 'competitor'] = complement_val

res['transfer_ratio'] = res.diff_from_worst / res.competitor
res['strike_ratio'] = res.competitor / diff_from_worst_to_best


res

# %%
res.plot(x='frac', y='val')

#%%
fig, ax = plt.subplots()
#res.plot(x='frac', y='val', ax=ax)
res.plot(x='frac', y='copy_ratio', ax=ax)
res.plot(x='frac', y='strike_ratio', ax=ax)
res.plot(x='frac', y='transfer_ratio', ax=ax)




# %%

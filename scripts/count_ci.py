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

df = pd.read_csv('../data/uber-raw-data-apr14.csv')
df[['Date', 'Time']] = df['Date/Time'].str.split(' ', expand=True)

#%%
df.head()

#%%

for small_frac in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,]:
    large_frac = 1 - small_frac
    small_samp = df.sample(frac=small_frac)
    large_samp = df.drop(small_samp.index)

    small_counts = small_samp.groupby('Date').Lat.count()
    small_widths = small_counts.apply(lambda x: confidence_width(x))

    large_counts = large_samp.groupby('Date').Lat.count()
    large_widths = large_counts.apply(lambda x: confidence_width(x))
    #print(counts)
    #print(widths)
    small_val = small_widths.mean() / small_frac
    print(small_frac, small_val)
    row_dicts.append({
        'frac': small_frac,
        'val': small_val,
        'company': 'small'
    })

    
    large_val = large_widths.mean() / large_frac
    row_dicts.append({
        'frac': small_frac,
        'val': large_val,
        'company': 'large'
    })

worst = row_dicts[0]['val']
best = row_dicts[-1]['val']
res = pd.DataFrame(row_dicts)
res.to_csv('count_ci_rows.csv', index=None)


# %%

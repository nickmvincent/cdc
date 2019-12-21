
#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
#dataset = 'pinterest-20'
dataset = 'inference'
#dataset = 'breast_cancer'
if dataset == 'inference':
    filename = 'inference_rows.csv'
    col = 'val'
    goal = 'minimize'

elif dataset == 'pinterest-20':
    filename = './RecSys2019/itemknn_pinterest_rows.csv'
    col = 'hitrate5'
    goal = 'maximize'

elif dataset == 'breast_cancer':
    filename = 'breast_cancer_rows.csv'
    col = 'hidden_score'
    goal = 'maximize'

df = pd.read_csv(filename)
if goal == 'minimize':
    worst = df[col].max()
    best = df[col].min()
else:
    worst = df[col].min()
    best = df[col].max()
df.head(3)

#%%
df


#%%
# agg different seeds across each frac / company combo
tmp = df.groupby(['frac', 'company']).mean()
# unstack and grab column of interest only
df = tmp.unstack()[col]
df['frac'] = df.index

df.head(3)


#%%
df
# %%
df['diff_small'] = worst - df['small']
df['diff_large'] = worst - df['large']

max_diff = worst - best

df['copy_ratio'] = df.diff_small/ max_diff


df['transfer_ratio'] = df.diff_small / df.diff_large
df['strike_ratio'] = 1 - df.diff_large / max_diff
df.head(3)

# %%
fig, ax = plt.subplots()

df.plot(x='frac', y='small', ax=ax, label='small')
df.plot(x='frac', y='large', ax=ax)


#%%
fig, ax = plt.subplots()
df.plot(x='frac', y='copy_ratio', ax=ax)
df.plot(x='frac', y='strike_ratio', ax=ax)
df.plot(x='frac', y='transfer_ratio', ax=ax)


#%%
df['weighted_copy_ratio'] = df['copy_ratio'] * df['frac']
fig, ax = plt.subplots()
#df.plot(x='frac', y='val', ax=ax)
df.plot(x='frac', y='weighted_copy_ratio', ax=ax)

# %%
tmp = df.copy_ratio.diff() / df.frac.diff()
tmp.plot()

# %%
df.frac.diff()

# %%

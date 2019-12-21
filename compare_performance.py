
#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
dataset = 'pinterest-20'
#dataset = 'inference'
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
baselines
worst = 0.165637

#%%
baselines = df[df.frac.isna()]
df = df.drop(baselines.index)
print(baselines)
print('===')
df

#%%
baselines

#%%
complements = {}
for frac_val in df.frac:
    complements[frac_val] = 1 - frac_val
flip = {
    'small': 'large',
    'large': 'small'
}
complement_df = df.copy()
complement_df.loc[:, 'company'] = complement_df.company.map(flip)
complement_df.loc[:, 'frac'] = complement_df.frac.map(complements)
complement_df['complement'] = True
complement_df

#%%
df_merged = pd.concat([df, complement_df]).reset_index(drop=True)
df_merged


#%%
# agg different seeds across each frac / company combo
tmp = df_merged.groupby(['frac', 'company']).mean()
tmp
#%%
# unstack and grab column of interest only
rdy = tmp.unstack()[col]
rdy['frac'] = rdy.index
rdy


# %%
# iowc = improvement over worst-case
# e.g. small's accuracy is 70% and random is 50%, so iowc is 20%
rdy['small_iowc'] = rdy['small'] - worst
rdy['large_iowc'] = rdy['large'] - worst

rdy['large_iobc'] = rdy['large'] - best

# max improvement over worst-case
max_iowc = best - worst

rdy['duplication'] = rdy.small_iowc / max_iowc
rdy['transfer'] = rdy.small_iowc / rdy.large_iowc

rdy.head(3)

#%%
rdy['deletion'] = worst / rdy.large_iowc
#rdy['deletion'] = rdy.large_iobc / 
rdy

#%%
worst

#%%
fig, ax = plt.subplots()
rdy.plot(x='frac', y='duplication', ax=ax)
rdy.plot(x='frac', y='deletion', ax=ax)
rdy.plot(x='frac', y='transfer', ax=ax)

# %%
fig, ax = plt.subplots()

rdy.plot(x='frac', y='small', ax=ax, label='small')
rdy.plot(x='frac', y='large', ax=ax)




#%%
rdy

#%%
baselines


#%%
if False:
    rdy['weighted_copy_ratio'] = rdy['copy_ratio'] * rdy['frac']
    fig, ax = plt.subplots()
    #rdy.plot(x='frac', y='val', ax=ax)
    rdy.plot(x='frac', y='weighted_copy_ratio', ax=ax)


# %%


#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
datasets = [
    #'pinterest-20',
    #'inference',
    'breast_cancer',
    #'ml-10m',
    #'cifar10',
    #'toxic'
]
rdy_dfs = {}
for dataset in datasets:
    print(dataset)
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
    elif dataset == 'ml-10m':
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
    print('df head 3\n', df.head(3))

    baselines = df[df.frac.isna()]
    df = df.drop(baselines.index)
    print('baselines\n', baselines, '\n===')


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
    print('complement_df head\n', complement_df.head(3), '\n===')

    df_merged = pd.concat([df, complement_df], sort=False).reset_index(drop=True)
    df_merged = df_merged.sort_values(by=['seed', 'company', 'frac'])
    df_merged['p_change'] = df_merged[col].diff()


    # agg different seeds across each frac / company combo
    tmp = df_merged.groupby(['frac', 'company']).mean()
    print('tmp', tmp, '\n===')
    # unstack and grab column of interest only
    rdy = tmp.unstack()[col]
    rdy['frac'] = rdy.index
    
    #df_merged[(df_merged.company == 'small') & (df_merged.p_change <0)]

    # iowc = improvement over worst-case
    # e.g. small's accuracy is 70% and random is 50%, so iowc is 20%
    # bad names here. "over" is "overloaded"
    rdy['small_iow'] = rdy['small'] - worst
    rdy['large_iow'] = rdy['large'] - worst

    rdy['small_iob'] = rdy['small'] - best
    rdy['large_iob'] = rdy['large'] - best

    # max improvement over worst-case
    max_iow = best - worst

    rdy['duplication'] = rdy.small_iow / max_iow
    rdy['transfer'] = rdy.small_iow / rdy.large_iow
    rdy['transfer'][rdy.frac > 0.5] = float('nan')
    rdy['deletion'] = worst / rdy.large_iow

    print('rdy\n', rdy.head(3))

    rdy['transfer_bonus'] = rdy.transfer - rdy.duplication
    rdy['dup_bonus'] = rdy.duplication - rdy.deletion

    fig, ax = plt.subplots()
    rdy.plot(x='frac', y='transfer_bonus', ax=ax)

    # fig, ax = plt.subplots()
    # rdy.plot(x='frac', y='dup_bonus', ax=ax)

    # fig, ax = plt.subplots()
    # rdy.plot(x='frac', y='small_iow', ax=ax)
    # rdy.plot(x='frac', y='large_iow', ax=ax)
    # plt.axhline(max_iow)

    rdy_dfs[dataset] = rdy

    if False:
        # sanity check - can see if this looks the same as prev
        cols = [
            'small_iow', 'large_iow', 
        ]
        for col in cols:
            rdy.loc[:, f'norm_{col}'] = rdy[col] / max_iow

        rdy['norm_large_over_small'] = rdy['small_iow'] / rdy['large_iow']


        fig, ax = plt.subplots()
        rdy.plot(x='frac', y='norm_small_iow', ax=ax)
        rdy.plot(x='frac', y='norm_large_iow', ax=ax)
        rdy.plot(x='frac', y='norm_large_over_small', ax=ax)



# %%
rdy_dfs.keys()


# %%
nrows = len(rdy_dfs)
print(nrows)
fig, ax = plt.subplots(nrows, 2)

#_, metrics_ax = plt.subplots(nrows, 1)

for i, (k, v) in enumerate(rdy_dfs.items()):
    v.plot(x='frac', y='small', ax=ax[i, 0], label='small')
    v.plot(x='frac', y='large', ax=ax[i, 0], label='large')
    v.plot(x='frac', y='duplication', ax=ax[i, 1])
    v.plot(x='frac', y='deletion', ax=ax[i, 1])
    v.plot(x='frac', y='transfer', ax=ax[i, 1])

    


# %%

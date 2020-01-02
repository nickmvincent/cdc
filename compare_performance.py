
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
datasets = [
    'pinterest-20',
    'count_ci',
    #'breast_cancer',
    'ml-10m',
    #'ml-10m_32_50',
    'cifar10',
    'toxic'
]
dataset_cols = {
    'pinterest-20': 'Hit Rate @ 5',
    'count_ci': 'Poisson CI',
    #'breast_cancer': 'Test Accuracy',
    'ml-10m': 'RMSE',
    'ml-10m_32_50': 'RMSE',
    'cifar10': 'Test Accuracy',
    'toxic': 'AUR'
}
rdy_dfs = {}
achieves_at_rows = []
transfer_bonus_dfs = []
for dataset in datasets:
    print(dataset)
    filename = f'results/{dataset}_rows.csv'
    if dataset == 'count_ci':
        col = 'val'
        goal = 'minimize'
    elif dataset == 'pinterest-20':
        filename = './RecSys2019/itemknn_pinterest_rows.csv'
        col = 'hitrate5'
        goal = 'maximize'
    elif dataset == 'breast_cancer':
        filename = 'results/breast_cancer_rows.csv'
        col = 'hidden_score'
        goal = 'maximize'
    elif dataset == 'cifar10':
        filename = 'results/cifar_rows.csv'
        col = 'valid_acc'
        goal = 'maximize'
    elif dataset == 'ml-10m' or dataset == 'ml-10m_32_50':
        col = 'rmse'
        goal = 'minimize'
    elif dataset == 'toxic':
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

    print('best and worst', best, worst, '\n===')


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

    # rdy['small_iob'] = rdy['small'] - best
    # rdy['large_iob'] = rdy['large'] - best

    # max improvement over worst-case
    max_iow = best - worst

    rdy['duplication'] = rdy.small_iow / max_iow
    rdy['transfer'] = rdy.small_iow / rdy.large_iow
    rdy['transfer'][rdy.frac > 0.5] = float('nan')
    rdy['deletion'] = 1 - rdy.large_iow / max_iow

    print('rdy\n', rdy.head(3))

    rdy['transfer_bonus'] = rdy.transfer - rdy.duplication
    rdy['dup_bonus'] = rdy.duplication - rdy.deletion

    fig, ax = plt.subplots()
    rdy.plot(x='frac', y='transfer_bonus', ax=ax)


    rdy_dfs[dataset] = rdy

    row = {
        'dataset': dataset,
    }
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        val = rdy[rdy.duplication > thresh].frac.min()
        row[f'achieves_{thresh}'] = val
        # achieves_at_rows.append({
        #     'dataset': dataset,
        #     'thresh': thresh,
        #     'achieves_at': achieves_at
        # })
        # print('achieves_at', achieves_at)
    achieves_at_rows.append(row)

    tmp_transfer = rdy[rdy.frac <= 0.5]
    tmp_transfer['dataset'] = dataset
    transfer_bonus_dfs.append(
        tmp_transfer[['dataset', 'transfer_bonus']]
    )
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

#%%
transfer_bonus_df = pd.concat(transfer_bonus_dfs, sort=False)
transfer_bonus_df.groupby('frac').transfer_bonus.max()

#%%
achieves_at_df = pd.DataFrame(achieves_at_rows)
achieves_at_df




# %%
sns.set_style('whitegrid')
nrows = len(rdy_dfs)
#nrows = 2
fig, ax = plt.subplots(nrows, 2, figsize=(6.5, 8), sharex=True)

#_, metrics_ax = plt.subplots(nrows, 1)

for i, (k, v) in enumerate(rdy_dfs.items()):
    v.plot(x='frac', y='small', ax=ax[i, 0], label='small', color='r')
    v.plot(x='frac', y='large', ax=ax[i, 0], label='large', color='k')
    v.plot(x='frac', y='duplication', ax=ax[i, 1])
    #v.plot(x='frac', y='deletion', ax=ax[i, 1])
    v.plot(x='frac', y='transfer', ax=ax[i, 1])
    ax[i, 0].set_ylabel(dataset_cols[k])
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax[i, 1].set_ylabel('PI Ratio')
    ax[i, 0].set_title(f'{k} Performance')
    ax[i, 1].set_title(f'{k} CDC Effectivness')

    ax[i, 1].text(0.5, 0.5, round(v['transfer_bonus'].max(), 2))
    print(k, round(v['transfer_bonus'].max(), 2))

    if i != 0:
        ax[i, 0].get_legend().remove()
        ax[i, 1].get_legend().remove()
    #plt.suptitle('CDC Simulations')
plt.savefig('reports/performance.png', dpi=300)

#%%
print(nrows)
print(len(rdy_dfs))
fig, ax = plt.subplots(nrows, 1, figsize=(6, 6), sharex=True)
for i, (k, v) in enumerate(rdy_dfs.items()):
    v.plot(x='frac', y='transfer_bonus', ax=ax[i])
    # #v.plot(x='frac', y='deletion', ax=ax[i, 1])
    # ax[i, 0].set_ylabel(dataset_cols[k])
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # ax[i, 1].set_ylabel('PI Ratio')
    ax[i].set_title(f'{k} Transfer Bonus')
    ax[i].set_xlim(0, 0.5)
    # ax[i, 1].set_title(f'{k} CDC Effectivness')

    # ax[i, 1].text(0.5, 0.5, round(v['transfer_bonus'].max(), 2))
    # print(k, round(v['transfer_bonus'].max(), 2))

    # if i != 0:
    #     ax[i, 0].get_legend().remove()
    #     ax[i, 1].get_legend().remove()
    #plt.suptitle('CDC Simulations')
plt.savefig('reports/transfer_bonus.png', dpi=300)

    



# %%
rdy = rdy_dfs['ml-10m']
fig, ax = plt.subplots()
rdy.plot(x='frac', y='small_iow', ax=ax)
rdy.plot(x='frac', y='large_iow', ax=ax)
rdy.transfer_bonus
#plt.axhline(max_iow)



# %%

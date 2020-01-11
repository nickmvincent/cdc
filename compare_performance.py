
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
datasets = [
    'pinterest-20',
    'ml-10m_32_50',
    #'ml-10m_64_100',
    #'ml-10m_2x',
    'cifar10',
    'toxic',

    #'count_ci',
    #'breast_cancer',
]
dataset_cols = {
    'pinterest-20': 'Hit Rate @ 5',
    'count_ci': 'Poisson CI',
    #'breast_cancer': 'Test Accuracy',
    'ml-10m': 'RMSE',
    'ml-10m_32_50': 'RMSE',
    'ml-10m_64_100': 'RMSE',
    'ml-10m_2x': 'RMSE',
    'cifar10': 'Test Accuracy',
    'toxic': 'AUR'
}

dataset_nice = {
    'pinterest-20': 'Pinterest RecSys',
    'count_ci': 'CI',
    #'breast_cancer': 'Test Accuracy',
    'ml-10m_32_50': 'ML-10M RecSys',
    'ml-10m_64_100': 'ML-10M RecSys',
    'ml-10m_2x': 'ML-10M RecSys 2x',
    'cifar10': 'CIFAR10',
    'toxic': 'Toxic Comments'
}
rdy_dfs = {}
valid_dfs = {}
achieves_at_rows = []
transfer_bonus_dfs = []
for dataset in datasets:
    print(dataset)
    filename = f'results/{dataset}_rows.csv'
    if dataset == 'count_ci':
        col = 'val'
        goal = 'minimize'
    elif dataset == 'breast_cancer':
        filename = 'results/breast_cancer_rows.csv'
        col = 'hidden_score'
        goal = 'maximize'

    # Primary experiments
    elif dataset == 'pinterest-20':
        filename = './RecSys2019/itemknn_pinterest_rows.csv'
        col = 'hitrate5'
        valid_col = 'hitrate5'
        goal = 'maximize'
    elif dataset == 'cifar10':
        col = 'test_acc'
        valid_col = 'valid_acc'
        goal = 'maximize'
    elif dataset == 'ml-10m' or dataset == 'ml-10m_32_50':
        col = 'hidden_rmse'
        valid_col = 'valid_rmse'
        goal = 'minimize'
    elif dataset == 'toxic':
        col = 'hidden_aur'
        valid_col = 'valid_aur'
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

    if dataset == 'pinterest-20':
        best = baselines[col].max()
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

    std = df_merged.groupby(['frac', 'company']).std().unstack()[col]
    print('tmp', tmp, '\n===')
    # unstack and grab column of interest only
    rdy = tmp.unstack()[col]
    try:
        valid_df = tmp.unstack()[valid_col]
        valid_df['frac'] = valid_df.index
        valid_std = df_merged.groupby(['frac', 'company']).std().unstack()[valid_col]
        valid_df['small_std'] = valid_std['small']
        valid_df['large_std'] = valid_std['large']
        valid_dfs[dataset] = valid_df
    except Exception as e:
        print(e)
    rdy['frac'] = rdy.index

    rdy['small_std'] = std['small']
    rdy['large_std'] = std['large']
    
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
        #'dataset': dataset,
        'Dataset': dataset_nice[dataset]
    }
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        val = rdy[rdy.duplication > thresh].frac.min()
        row[f'{thresh}'] = val
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
achieves_at_df = pd.DataFrame(achieves_at_rows).sort_values('0.5')[[
    'Dataset', '0.5', '0.6', '0.7', '0.8', '0.9'
]]
achieves_at_df.to_csv('achieves_at.csv', index=False)

achieves_at_df




# %%
sns.set_style('whitegrid')
nrows = len(rdy_dfs)
#nrows = 2
fig, ax = plt.subplots(nrows, 2, figsize=(7, 7), sharex=True)

#_, metrics_ax = plt.subplots(nrows, 1)

for i, (k, v) in enumerate(rdy_dfs.items()):
    print(v['small_std'])
    nice_name = dataset_nice[k]
    v.plot(x='frac', y='small', yerr='small_std', ax=ax[i, 0], label='small', color='k', marker='o')
    v.plot(x='frac', y='large', yerr='large_std', ax=ax[i, 0], label='large', color='r', marker='x')
    v.plot(x='frac', y='duplication', ax=ax[i, 1], color='k', marker='o')
    v.plot(x='frac', y='transfer', ax=ax[i, 1], color='r', marker='x')

    ax[i, 0].set_xlabel('Group Size')
    ax[i, 1].set_xlabel('Group Size')

    ax[i, 0].set_ylabel(dataset_cols[k])
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax[i, 1].set_ylabel('PIR')
    ax[i, 0].set_title(f'{nice_name}\nPerformance')
    ax[i, 1].set_title(f'{nice_name}\nCDC Effectiveness')


    # UNCOMMENT TO SHOW TRANSFER BONUS ON PLOTS
    #ax[i, 1].text(0.5, 0.5, round(v['transfer_bonus'].max(), 2))
    print(k, round(v['transfer_bonus'].max(), 2))

    if i != 0:
        ax[i, 0].get_legend().remove()
        ax[i, 1].get_legend().remove()
    #plt.suptitle('CDC Simulations')
plt.savefig('reports/performance.png', dpi=300)

#%%
print(nrows)
print(len(rdy_dfs))
fig, ax = plt.subplots(nrows, 1, figsize=(6, 6), sharex=True, sharey=True)
for i, (k, v) in enumerate(rdy_dfs.items()):
    v.plot(x='frac', y='transfer_bonus', ax=ax[i], marker='o')
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
fig, ax = plt.subplots()
valid_dfs['cifar10'].plot(x='frac', y='small', ax=ax)
#valid.plot(x='frac', y='large', ax=ax)
rdy_dfs['cifar10'].plot(x='frac', y='small', ax=ax)

# fig, ax = plt.subplots()
# rdy.plot(x='frac', y='small_iow', ax=ax)
# rdy.plot(x='frac', y='large_iow', ax=ax)
# rdy.transfer_bonus

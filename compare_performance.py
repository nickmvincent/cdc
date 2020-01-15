
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

# store a rdy-to-plot df for each hidden evaluation metric
hidden_dfs = {}
# a helper dict for storing 
# (1) when PIR achieves a certain threshold and (2) the transfer bonus
hidden_d = {
    'achieves_at': [],
    'transfer_bonus': [],
}

# store a rdy-to-plot df for each company-perspective evaluation
co_dfs = {}
co_d = {
    'achieves_at': [],
    'transfer_bonus': [],
}

# achieves_at_rows = []
# transfer_bonus_dfs = []
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
        hidden_col = 'weighted_hitrate5'
        co_col = 'hitrate5'
        goal = 'maximize'
    elif dataset == 'cifar10':
        hidden_col = 'test_acc'
        co_col = 'valid_acc'
        goal = 'maximize'
    elif dataset == 'ml-10m' or dataset == 'ml-10m_32_50':
        hidden_col = 'hidden_rmse'
        co_col = 'valid_rmse'
        goal = 'minimize'
    elif dataset == 'toxic':
        hidden_col = 'hidden_aur'
        co_col = 'valid_aur'
        goal = 'maximize'

    for dfs, rows_for_dfs, col, name in zip(
        [hidden_dfs, co_dfs],
        [hidden_d, co_d],
        [hidden_col, co_col],
        ['hidden', 'co'],
    ):
        df = pd.read_csv(filename)

        if dataset == 'pinterest-20':
            df.loc[:, 'weighted_hitrate5'] = df['hitrate5']
            mask1 = df.company == 'small'
            masked1 = df[mask1]
            mask2 = (df.company == 'large') & (~df.frac.isna())
            masked2 = df[mask2]
            TOP_POP_HITRATE = .1668
            df.loc[mask1, 'weighted_hitrate5'] = masked1.frac * masked1['hitrate5'] + (1-masked1.frac) * TOP_POP_HITRATE
            df.loc[mask2, 'weighted_hitrate5'] = (1-masked2.frac) * masked2['hitrate5'] + (masked2.frac) * TOP_POP_HITRATE
        
        if goal == 'minimize':
            worst = df[col].max()
            best = df[col].min()
        else:
            worst = df[col].min()
            best = df[col].max()

        #print('df head 3\n', df.head(3))

        baselines = df[df.frac.isna()]

        if dataset == 'pinterest-20':
            best = baselines[col].max()
            print(baselines)
        df = df.drop(baselines.index)
        #print('baselines\n', baselines, '\n===')
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
        #print('complement_df head\n', complement_df.head(3), '\n===')

        df_merged = pd.concat([df, complement_df], sort=False).reset_index(drop=True)
        df_merged = df_merged.sort_values(by=['seed', 'company', 'frac'])
        df_merged['p_change'] = df_merged[col].diff()

        # agg different seeds across each frac / company combo
        tmp = df_merged.groupby(['frac', 'company']).mean()

        std = df_merged.groupby(['frac', 'company']).std().unstack()[col]
        print('tmp', tmp, '\n===')
        # unstack and grab column of interest only
        rdy = tmp.unstack()[col]
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

        # fig, ax = plt.subplots()
        # rdy.plot(x='frac', y='transfer_bonus', ax=ax)
        dfs[dataset] = rdy

        row = {
            #'dataset': dataset,
            'Dataset': dataset_nice[dataset]
        }
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            val = rdy[rdy.duplication > thresh].frac.min()
            row[f'{thresh}'] = val
        rows_for_dfs['achieves_at'].append(row)

        tmp_transfer = rdy[rdy.frac <= 0.5]
        tmp_transfer.loc[:, 'dataset'] = dataset
        rows_for_dfs['transfer_bonus'].append(
            tmp_transfer[['dataset', 'transfer_bonus']]
        )
    for rows_for_dfs, name in zip(
        [hidden_d, co_d],
        ['hidden', 'co'],
    ):
        transfer_bonus_df = pd.concat(rows_for_dfs['transfer_bonus'], sort=True)
        transfer_bonus_df.to_csv(f'reports/transfer_bonus_{name}.csv', index=False)
        print('Max')
        print(transfer_bonus_df.groupby('frac').transfer_bonus.max())
        rows_for_dfs['transfer_bonus_df'] = transfer_bonus_df

        achieves_at_df = pd.DataFrame(rows_for_dfs['achieves_at']).sort_values('0.5')[[
            'Dataset', '0.5', '0.6', '0.7', '0.8', '0.9'
        ]]
        achieves_at_df.to_csv(f'reports/achieves_at_{name}.csv', index=False)




# %%
sns.set_style('whitegrid')
nrows = len(hidden_dfs)
#nrows = 2
fig, ax = plt.subplots(nrows, 2, figsize=(7, 8), sharex=True)

#_, metrics_ax = plt.subplots(nrows, 1)

for i, (k, v) in enumerate(hidden_dfs.items()):
    #print(v['small_std'])
    nice_name = dataset_nice[k]
    v.plot(x='frac', y='small', yerr='small_std', ax=ax[i, 0], label='SMALL', color='k', marker='o')
    v.plot(x='frac', y='large', yerr='large_std', ax=ax[i, 0], label='LARGE', color='r', marker='x')
    v.plot(x='frac', y='duplication', ax=ax[i, 1], color='k', marker='o', label='CDC only')
    v.plot(x='frac', y='transfer', ax=ax[i, 1], color='r', marker='x', label='CDC + deletion')

    # co_df = co_dfs[k]
    # co_df.plot(x='frac', y='small', yerr='small_std', ax=ax[i, 0], label='co small', color='g', marker='o', linestyle='None')
    # co_df.plot(x='frac', y='large', yerr='large_std', ax=ax[i, 0], label='co large', color='b', marker='x', linestyle='None')
    ax[i, 0].set_xlabel('Group Size')
    ax[i, 1].set_xlabel('Group Size')

    ax[i, 0].set_ylabel(dataset_cols[k])
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax[i, 1].set_ylabel('PIR')
    ax[i, 0].set_title(f'{nice_name}\nHidden Test Set Performance')
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
fig, ax = plt.subplots(nrows, 2, figsize=(7, 8), sharex=True)

#_, metrics_ax = plt.subplots(nrows, 1)

for i, (k, v) in enumerate(co_dfs.items()):
    nice_name = dataset_nice[k]
    v.plot(x='frac', y='small', yerr='small_std', ax=ax[i, 0], label='small', color='k', marker='o')
    v.plot(x='frac', y='large', yerr='large_std', ax=ax[i, 0], label='large', color='r', marker='x')
    v.plot(x='frac', y='duplication', ax=ax[i, 1], color='k', marker='o', label='CDC only')
    v.plot(x='frac', y='transfer', ax=ax[i, 1], color='r', marker='x', label='CDC + deletion')

    ax[i, 0].set_xlabel('Group Size')
    ax[i, 1].set_xlabel('Group Size')

    ax[i, 0].set_ylabel(dataset_cols[k])
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax[i, 1].set_ylabel('PIR')
    ax[i, 0].set_title(f'{nice_name}\nCompany-Perspective Performance')
    ax[i, 1].set_title(f'{nice_name}\nCDC Effectiveness')


    # UNCOMMENT TO SHOW TRANSFER BONUS ON PLOTS
    #ax[i, 1].text(0.5, 0.5, round(v['transfer_bonus'].max(), 2))
    print(k, round(v['transfer_bonus'].max(), 2))

    if i != 0:
        ax[i, 0].get_legend().remove()
        ax[i, 1].get_legend().remove()
    #plt.suptitle('CDC Simulations')
plt.savefig('reports/company_perspective_performance.png', dpi=300)

#%%
fig, ax = plt.subplots(nrows, 1, figsize=(6, 6), sharex=True, sharey=True)
for i, (k, v) in enumerate(hidden_dfs.items()):
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

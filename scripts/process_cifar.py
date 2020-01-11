#%%
import pandas as pd
import glob

folder = 'results/cifar10'
dfs = []
for path in glob.glob(folder + '/*'):
    df = pd.read_csv(path, index_col=0).dropna(axis=1,how='all')
    print(df.head())
    grouped = df.groupby(['company', 'frac'])
    for name, group in grouped:
        c = group.copy()
        test_acc_row = c.iloc[-1]
        c = c[(c.epoch == 24) & (c['test acc'] == 0)]
        c['test_acc'] = test_acc_row['test acc']
        print(test_acc_row['test acc'])
        print(c.head())
        print('==')
        dfs.append(c)

merged = pd.concat(dfs, sort=True)

# %%
merged[
    ['company', 'frac', 'seed', 'valid_acc', 'test_acc']
].to_csv('results/cifar10_rows.csv')

# %%

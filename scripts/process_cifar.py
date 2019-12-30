#%%
import pandas as pd
df = pd.read_csv('results/cifar_cdc_0.csv', index_col=0).reset_index()


# %%
df[df.epoch == 24][
    ['company', 'frac', 'seed', 'valid_acc']
].to_csv('results/cifar_rows.csv')

# %%

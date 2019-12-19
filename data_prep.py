# Save from Sklearn so we have the column names
# double check it though
#%%
import pandas as pd
from sklearn import datasets
import numpy as np

cancer = datasets.load_breast_cancer()
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                columns= np.append(cancer['feature_names'], ['target']))
df.to_csv('data/breast_cancer/sklearn_breast_cancer.csv', index=False)

# %%

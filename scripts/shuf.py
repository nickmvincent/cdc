#import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=3)

pre = './data/toxic'
with open(f'{pre}/train.csv') as f:
    lines = f.read().splitlines()
    X = np.array(lines)
    #print(X)
print('loaded X')
print(X.size)
for i, (train_index, test_index) in enumerate(kf.split(X)):
    X[train_index].tofile(f'{pre}/train{i}.csv', sep="\n", format='%s')
    X[test_index].tofile(f'{pre}/test{i}.csv', sep="\n", format='%s')
    break

#tr -d \' < train0.csv > train0.csv
#tr -d \' < test0.csv > test0.csv
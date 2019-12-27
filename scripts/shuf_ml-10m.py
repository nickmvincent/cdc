import numpy as np
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=3)

pre = '../libFM/libfm-1.4.2.src/'
with open(f'{pre}/data/ml-10m/ratings.dat') as f:
    lines = f.read().splitlines()
    X = np.array(lines)
    print(X)

for i, (train_index, test_index) in enumerate(kf.split(X)):
    X[train_index].tofile(f'{pre}/train{i}.csv', sep="\n", format='%s')
    X[test_index].tofile(f'{pre}/test{i}.csv', sep="\n", format='%s')
    break

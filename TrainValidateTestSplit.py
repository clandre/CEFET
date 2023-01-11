import pandas as pd 
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split

'''
path ='data_ecg/'
csv_files = glob.glob(os.path.join(path, "*.csv"))
list_columns = [f'a{i}' for i in range(187)]
list_columns.append('target')

for k,f in enumerate(csv_files):

    if k == 0:
        first = np.loadtxt(f"{f}",delimiter=",", dtype=float)
    else:
        other = np.loadtxt(f"{f}",delimiter=",", dtype=float)
        first = np.concatenate((first,other),axis=0)

df = pd.DataFrame(first, columns = list_columns)
df.to_parquet("AllData.parquet",index=False)
'''

df = pd.read_parquet("AllData.parquet")

x = df.iloc[:,:-1]
y = df['target'].copy()

x_train, x_rem, y_train, y_rem = train_test_split(x,y, train_size=0.7)

x_valid, x_test, y_valid, y_test = train_test_split(x_rem,y_rem, test_size=0.5)

x_train['target'] = y_train
x_valid['target'] = y_valid
x_test['target'] = y_test

print('Shape dos dados de treino: ',x_train.shape)
print('Shape dos dados de validação: ',x_valid.shape)
print('Shape dos dados de test: ',x_test.shape)


x_train.to_parquet('x_train.parquet',index=False)
x_valid.to_parquet('x_valid.parquet',index=False)
x_test.to_parquet('x_test.parquet',index=False)
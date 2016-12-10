import numpy as np
import pandas as pd
import scipy

from sklearn import preprocessing

df = pd.read_csv('./test_file.csv')

to_delete = []

x = df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

for i in range (0,len(df.columns)-1 ):
	for j in range(i+1, len(df.columns)-1):
		corr, value = scipy.stats.pearsonr(df[i], df[j])
		if(corr > 0.9 or corr*-1 > 0.9):
			to_delete.append(i)
			break
		
	

print to_delete



import pandas as pd
import numpy as np


df = pd.read_csv('dataset/colunas_apagadas_exclui_linhas_valor_ruim.csv')


indx_1 = df[df.TARGET == 1].index
quant = len(indx_1)


inds = df[df.TARGET == 0].index

result_1 = df.loc[indx_1]

# esses são os indices que vao ficar
sampled_indices = np.random.choice(inds, quant * 6, replace=False)

result_2 = df.loc[sampled_indices]

frames = [result_1, result_2]
result = pd.concat(frames)

zerocount = result[result.TARGET == 0].index

result.to_csv('dataset/undersampling_6.csv', index=False)
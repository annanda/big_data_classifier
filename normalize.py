import pandas as pd
import scipy

from sklearn import preprocessing

def colunas_apagar_pearson(df, train=1):
    to_delete = []
    keys_df = df.keys()

    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df.columns = keys_df[:]

    if train:
        range_colunas = len(df.columns)-1
    else:
        range_colunas = len(df.columns)
    for i in range(range_colunas):
        for j in range(i+1, range_colunas):
            corr, value = scipy.stats.pearsonr(df.ix[:,i:i+1], df.ix[:,j:j+1])
            if(corr > 0.9 or corr*-1 > 0.9):
                to_delete.extend(df.ix[:,i:i+1].keys())
                break

    return to_delete


def apaga_colunas_pearson(to_delete, df, df_name_export):
    for key in to_delete:
        df = df.drop(key, 1)

    df.to_csv('dataset/' + df_name_export)


if __name__ == '__main__':
    df = pd.read_csv('dataset/train_file.csv')
    df_name_export = 'colunas_apagadas_pearson_refatorada.csv'
    to_delete = colunas_apagar_pearson(df)
    apaga_colunas_pearson(to_delete, df, df_name_export)
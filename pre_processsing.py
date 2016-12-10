import pandas as pd
from sklearn.cluster import KMeans
import scipy
from sklearn import preprocessing


def elimina_colunas_mesmo_valor():
    data = pd.read_csv('dataset/train_file.csv')
    keys_data = data.keys()
    selected_keys = []
    d = {}
    for key_data in keys_data:
        contagem = data[key_data].value_counts()
        quantidade_valores = len(contagem)
        if quantidade_valores > 1 and (contagem.iloc[0] < 70015 and contagem.iloc[1] < 70015):
            d[key_data] = data[key_data].values
            selected_keys.append(key_data)
    new_data = pd.DataFrame(d)
    new_data.to_csv('dataset/train_colunas_limpas_nova.csv')
    selected_keys.remove('TARGET')
    return selected_keys


def elimina_colunas_test_set(selected_keys):
    test = pd.read_csv('dataset/test_file.csv')
    d = {}
    for key_data in selected_keys:
        d[key_data] = test[key_data].values
    new_data = pd.DataFrame(d)
    new_data.to_csv('dataset/test_colunas_limpas.csv')


def usando_k_means():
    dataset = pd.read_csv('dataset/train_colunas_limpas_2.csv')
    #apagando uma coluna fantasma de indices
    dataset = dataset.drop('Unnamed: 0', 1)
    test = pd.read_csv('dataset/test_colunas_limpas.csv')
    test = test.drop('Unnamed: 0', 1)

    x = dataset.values[:, :-1]

    x_test = test.values[:, :]
    kmeans = KMeans(n_clusters=2).fit_predict(x)
    kmeans_test = KMeans(n_clusters=3).fit_predict(x_test)

    dataset['kmeans_1'] = 0
    dataset['kmeans_2'] = 0

    test['kmeans_1'] = 0
    test['kmeans_2'] = 0

    for i, result in enumerate(kmeans):
        if result == 0:
            dataset['kmeans_1'][i] = 1
        elif result == 1:
            dataset['kmeans_2'][i] = 1

    for i, result in enumerate(kmeans_test):
        if result == 0:
            dataset['kmeans_1'][i] = 1
        elif result == 1:
            dataset['kmeans_2'][i] = 1

    dataset.to_csv('dataset/train_kmeans_2clusters.csv')
    test.to_csv('dataset/test_kmeans_2clusters.csv')


def elimina_exemplos_ruins():
    dataset = pd.read_csv('dataset/train_colunas_limpas_2.csv')
    dataset = dataset.drop('Unnamed: 0', 1)
    keys_data = dataset.keys()
    keys_data = list(keys_data)
    keys_data.remove('TARGET')
    for key_data in keys_data:
        media = dataset[key_data].mean()
        desvio_padrao = dataset[key_data].std()
        coluna = dataset[key_data]
        for i in range(len(coluna)):
            if coluna[i]:
                if (coluna[i] - media) > desvio_padrao:
                    print("Linha: {}, valor da linha: {}\n".format(i, coluna[i], key_data))
                    print("chave: {}, media: {}, desvio_padrao: {}\n\n".format( key_data, media, desvio_padrao))
                    dataset.drop(dataset.index[[i]])
    dataset.to_csv('dataset/train_limpando_linhas.csv')


def descobre_colunas_mesmo_valor(df):
    keys_data = df.keys()
    to_delete = []
    for key_data in keys_data:
        contagem = df[key_data].value_counts()
        quantidade_valores = len(contagem)
        if quantidade_valores > 1 and (contagem.iloc[0] > 70015 or contagem.iloc[1] > 70015):
            to_delete.append(key_data)
    return to_delete


def descobre_colunas_pearson(df, train=1):
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


def apaga_colunas(list_to_delete, df, df_name_export):
    for key in list_to_delete:
        df = df.drop(key, 1)

    df.to_csv('dataset/' + df_name_export)


if __name__ == '__main__':
    df = pd.read_csv('dataset/train_file.csv')
    df_name_export = 'colunas_apagadas_pearson_refatorada.csv'
    to_delete = descobre_colunas_pearson(df)
    print('To delete: {}'.format(to_delete))
    apaga_colunas(to_delete, df, df_name_export)

    df_2 = pd.read_csv('dataset/colunas_apagadas_pearson_refatorada.csv')
    df_2.drop('Unnamed: 0', 1)

    to_delete_2 = descobre_colunas_mesmo_valor(df_2)
    print('To delete 2: {}'.format(to_delete_2))

    apaga_colunas(to_delete_2, df_2, 'colunas_apagadas_mesmo_valor.csv')

    test = pd.read_csv('dataset/test_file.csv')
    to_delete_test = to_delete + to_delete_2
    print('to delete test')
    print(to_delete_test)
    apaga_colunas(to_delete_test, test, 'teste_processado.csv')

import pandas as pd
from sklearn.cluster import KMeans


def eliminando_colunas_mesmo_valor():
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
    new_data.to_csv('dataset/train_colunas_limpas_2.csv')
    selected_keys.remove('TARGET')
    return selected_keys

def eliminando_colunas_test_set(selected_keys):
    test = pd.read_csv('dataset/test_file.csv')
    d = {}
    for key_data in selected_keys:
        d[key_data] = test[key_data].values
    new_data = pd.DataFrame(d)
    new_data.to_csv('dataset/test_colunas_limpas.csv')

# def correlacao_pearson():
#     data = pd.read_csv('dataset/train_colunas_limpas_2.csv')
#     data.T.corr(method='spearman')

def usando_k_means():
    dataset = pd.read_csv('dataset/train_colunas_limpas_2.csv')
    test = pd.read_csv('dataset/test_colunas_limpas.csv')

    x = dataset.values[:, :-1]
    y = dataset['TARGET']

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

if __name__ == '__main__':
    # selectec_keys = eliminando_colunas_mesmo_valor()
    # eliminando_colunas_test_set(selectec_keys)
    usando_k_means()
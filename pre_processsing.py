import pandas as pd
import numpy as np


def eliminando_colunas_mesmo_valor():
    data = pd.read_csv('dataset/train_file.csv')
    keys_data = data.keys()
    d = {}
    for key_data in keys_data:
        contagem = data[key_data].value_counts()
        quantidade_valores = len(contagem)
        if quantidade_valores > 1 and (contagem.iloc[0] < 70015 and contagem.iloc[1] < 70015):
            d[key_data] = data[key_data].values

    # diminuiu de 370 colunas para 336 colunas
    new_data = pd.DataFrame(d)
    new_data.to_csv('dataset/train_colunas_limpas_2.csv')


def eliminando_outliers():
    data = pd.read_csv('dataset/train_colunas_limpas_2.csv')
    print(data)
    # keys_data = data.keys()
    # d = {}
    # for key_data in keys_data:
    #     quantidade_valor = len(data[key_data].value_counts())
    #     if quantidade_valor == 2:
    #         print("Key: {}, valores: {}".format(key_data, data[key_data].value_counts()))


if __name__ == '__main__':
    eliminando_colunas_mesmo_valor()
    eliminando_outliers()
import pandas as pd
import numpy as np


def eliminando_colunas_mesmo_valor():
    data = pd.read_csv('dataset/train_file.csv')
    keys_data = data.keys()
    d = {}
    for key_data in keys_data:
        quantidade_valor = len(data[key_data].value_counts())
        if quantidade_valor != 1:
            d[key_data] = data[key_data].values
        if quantidade_valor == 2:
            print("Key: {}, valores: {}".format(key_data, data[key_data].value_counts()))

    # diminuiu de 370 colunas para 336 colunas
    new_data = pd.DataFrame(d)
    print(new_data)


if __name__ == '__main__':
    eliminando_colunas_mesmo_valor()
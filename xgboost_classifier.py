import pandas as pd
import xgboost as xgb



# Load the data
train_df = pd.read_csv('dataset/colunas_apagadas_exclui_linhas_valor_ruim.csv')
# train_df = pd.read_csv('dataset/undersampling_2-5.csv')
test_df = pd.read_csv('dataset/teste_processado.csv')


# Prepare the inputs for the model
train_X = train_df.values[:,:-1]
test_X = test_df.values[:,:]
train_y = train_df['TARGET']

# dtrain = xgb.DMatrix(train_X)
# dtest = xgb.DMatrix(test_X)

# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
gbm = xgb.XGBClassifier(max_depth=25, n_estimators=300, learning_rate=0.001).fit(train_X, train_y)
predictions = gbm.predict_proba(test_X)
# param = {'max_depth':2, 'eta':0.3, 'silent':1, 'objective':'binary:logistic' }
# num_round = 2
# bst = xgb.train(param, train_X, num_round)
# preds = bst.predict_proba(test_X)
# print(preds[1])

with open('resultados/xgb_9.txt', 'w') as file:
    for line in predictions:
        linha = line[1]
        porcentagem = str(linha) + '\n'
        file.write(porcentagem)
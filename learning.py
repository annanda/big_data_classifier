import pandas as pd
from sklearn.cross_validation import cross_val_score, cross_val_predict
# from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

def try_classifier(clf, tag):
    dataset = pd.read_csv('dataset/colunas_apagadas_mesmo_valor_exclui_linhas_menor_zero.csv')
    test = pd.read_csv('dataset/teste_processado.csv')

    x = dataset.values[:,:-1]
    y = dataset['TARGET']

    x_test = test.values[:,:]

    clf.fit(x, y)
    predictions = clf.predict_proba(x_test)

    scores = cross_val_score(clf, x, y, scoring='roc_auc')
    print("Auc (%s): %0.3f (+/- %0.3f)" % (tag, scores.mean(), scores.std() * 2))

    with open( tag + '.txt', 'w') as file:
        for line in predictions:
            file.write(str(line[1]) + '\n')


if __name__ == '__main__':
    try_classifier(GradientBoostingClassifier(), 'gb_linhas_excluidas')
    try_classifier(RandomForestClassifier(), 'random_linhas_excluidas')
    try_classifier(ExtraTreesClassifier(), 'extra_tree_linhas_excluidas')
    try_classifier(DecisionTreeClassifier(), 'decition_linhas_excluidas')
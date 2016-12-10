import pandas as pd
from sklearn.cross_validation import cross_val_score, cross_val_predict
# from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

def try_classifier(clf, tag):
    dataset = pd.read_csv('dataset/train_limpando_linhas.csv')
    test = pd.read_csv('dataset/test_colunas_limpas.csv')

    x = dataset.values[:,:-1]
    y = dataset['TARGET']

    x_test = test.values[:,:]

    clf.fit(x, y)
    predictions = clf.predict_proba(x_test)

    scores = cross_val_score(clf, x, y, scoring='roc_auc')
    print("Auc (%s): %0.3f (+/- %0.3f)" % (tag, scores.mean(), scores.std() * 2))

    with open('random_forest.txt', 'w') as file:
        for line in predictions:
            file.write(str(line[1]) + '\n')


if __name__ == '__main__':
    try_classifier(RandomForestClassifier(), 'random forest')
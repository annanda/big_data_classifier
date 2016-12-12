import pandas as pd
# from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

def try_classifier(clf, tag):
    dataset = pd.read_csv('dataset/undersampling_6.csv')
    test = pd.read_csv('dataset/teste_processado.csv')

    x = dataset.values[:,:-1]
    y = dataset['TARGET']

    x_test = test.values[:,:]

    clf.fit(x, y)
    predictions = clf.predict_proba(x_test)

    scores = cross_val_score(clf, x, y, cv=5, scoring='roc_auc')
    print("Auc (%s): %0.3f (+/- %0.3f)" % (tag, scores.mean(), scores.std() * 2))

    with open('resultados/' + tag + '.txt', 'w') as file:
        for line in predictions:
            file.write(str(line[1]) + '\n')


if __name__ == '__main__':
    # try_classifier(GradientBoostingClassifier(min_samples_split=3), 'gradientboosting_min_samples_split_3')
    try_classifier(GradientBoostingClassifier(), 'gb_undersampling_6')
    # try_classifier(RandomForestClassifier(), 'random_linhas_excluidas_menor_zero')
    # try_classifier(ExtraTreesClassifier(), 'extra_tree_linhas_excluidas_menor_zero')
    # try_classifier(DecisionTreeClassifier(), 'decition_linhas_excluidas_menor_zero')
    # try_classifier(GradientBoostingClassifier(max_features=1), 'gradientboosting_max_features_1')
    # try_classifier(GradientBoostingClassifier(max_features=3), 'gradientboosting_max_features_3')
    # try_classifier(GradientBoostingClassifier(max_features='auto'), 'gradientboosting_max_features_auto')
    try_classifier(GradientBoostingClassifier(max_features='sqrt'), 'gradientboosting_max_features_sqrt_undersampling_6')
    # try_classifier(GradientBoostingClassifier(max_features='log2'), 'gradientboosting_max_features_log2')
    # try_classifier(GradientBoostingClassifier(max_features=None), 'gradientboosting_max_features_None')
    # try_classifier(GradientBoostingClassifier(n_estimators=1000), 'gradientboosting_n_estimators_1000')
    # try_classifier(GradientBoostingClassifier(n_estimators=3000), 'gradientboosting_n_estimators_3000')
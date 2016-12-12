# Relatorio

1. Eliminar as colunas cujo conteúdo das células é o mesmo
    diminuiu de 370 colunas para 336 colunas

2. Eliminar as colunas cujo conteúdo das células é o mesmo para até 70015 linhas
    diminuiu de 336 colunas para 276 colunas



96,04% dos exemplos são da classe 0
3,95% dos exemplos são da classe 1

Eliminar ate 5 valores diferentes numa coluna é melhor que eliminar apenas as colunas com todos os valores iguais.


Auc (random_forest): 0.677 (+/- 0.012)
Auc (extra_tree_classifier): 0.631 (+/- 0.006)
Auc (decition_tree): 0.569 (+/- 0.006)

Eliminar ate 15 valores diferentes numa coluna é pior que eliminar apenas as colunas com os valores até 5 diferentes.

Auc (gb_menos_15): 0.836 (+/- 0.005)
Auc (random_forest_menos_15): 0.678 (+/- 0.003)
Auc (extra_tree_classifier_menos_15): 0.643 (+/- 0.014)
Auc (decition_tree_menos_15): 0.569 (+/- 0.001)


excluindo as linhas de valores ruins da coluna var3:
não melhorou

excluindo as linhas de valores ruins da coluna var3, se target não for 1.
houve uma melhora
De 0.840261 para 0.841493.

Auc (gradientboosting_min_samples_split_3): 0.836 (+/- 0.009)

Auc (gradientboosting_max_features_1): 0.783 (+/- 0.016)
Auc (gradientboosting_max_features_3): 0.805 (+/- 0.005)
Auc (gradientboosting_max_features_auto): 0.835 (+/- 0.009)
Auc (gradientboosting_max_features_sqrt): 0.832 (+/- 0.005)  --> melhor resultado
Auc (gradientboosting_max_features_log2): 0.818 (+/- 0.016)


Rede Neural:
Não consegui =(

Undersampling 50/50 -> Não melhorou
Auc (gb_undersampling): 0.828 (+/- 0.024)
Auc (gradientboosting_max_features_sqrt_undersampling): 0.824 (+/- 0.025)

Melhor resultado foi com undersampling, pegando quantidade de exemplos 0 igual a 2.5 vezes o numero de exemplos 1.
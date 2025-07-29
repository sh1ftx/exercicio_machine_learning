import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Criação de um dataset fictício para classificação
X, y = make_classification(
    n_samples=5000,
    n_features=8,
    n_informative=5,
    n_redundant=2,
    n_classes=2
)

# Objeto de validação cruzada estratificada com 5 divisões (folds)
cv = StratifiedKFold(n_splits=5)

# Instancia o classificador Random Forest
model = RandomForestClassifier()

# Validação cruzada padrão com cross_validate
cv_results = cross_validate(model, X, y, cv=cv)

# Impressão dos resultados da validação cruzada
print("Keys do dict automático: ", sorted(cv_results.keys()))
print("Resultados por fold: ", cv_results['test_score'])
print("Média: ", cv_results['test_score'].mean())
print("Desvio padrão: ", cv_results['test_score'].std())

# Grid de parâmetros a serem testados no modelo (n_estimators e criterion)
param_grid = {
    'n_estimators': [50, 100, 300, 500],
    'criterion': ["gini", "entropy", "log_loss"]
}

# Busca exaustiva de hiperparâmetros com validação cruzada
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    refit=True
)

# Execução da busca pelos melhores parâmetros
grid_search.fit(X, y)

# Impressão da melhor combinação de parâmetros encontrada
print("Melhores parâmetros encontrados:", grid_search.best_params_)

# Uso do melhor modelo encontrado após o GridSearchCV
melhor_modelo = grid_search.best_estimator_

# Avaliação final do modelo com os melhores parâmetros em nova validação cruzada
final_results = cross_validate(melhor_modelo, X, y, cv=cv)

# Impressão dos resultados finais com o modelo otimizado
print("Resultados finais por fold: ", final_results['test_score'])
print("Média final: ", final_results['test_score'].mean())
print("Desvio padrão final: ", final_results['test_score'].std())

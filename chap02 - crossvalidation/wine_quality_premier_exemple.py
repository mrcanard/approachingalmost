import pandas as pd

df = pd.read_csv("chap02 - crossvalidation/winequality-red.csv")

# Mapping de la quality de 3-8 vers 0-5
quality_mapping = {
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5
}

df.loc[:, "quality"] = df.quality.map(quality_mapping)

# data splitting

## mélange
df = df.sample(frac=1).reset_index(drop=True)

## top 1000 for training
df_train = df.head(1000)

## botton 599 for test
df_test = df.tail(599)

# Decision tree pour la classification
from sklearn import tree
from sklearn import metrics

## décision tree de profondeur 3
clf = tree.DecisionTreeClassifier(max_depth=3)

## Selection des colonnes d'entrainement
cols = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]

## Entrainement du modèle
clf.fit(df_train[cols], df_train.quality)

# Test de la précisi on de la prédiction
## sur le training set
train_predictions = clf.predict(df_train[cols])

## sur le test set
test_predictions = clf.predict(df_test[cols])

## Précision sur le training set
metrics.accuracy_score(df_train.quality, train_predictions)

## Précision sur le training set
metrics.accuracy_score(df_test.quality, test_predictions)

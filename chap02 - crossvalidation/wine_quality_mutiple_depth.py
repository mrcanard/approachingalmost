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

# Affichage
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Label text
matplotlib.rc("xtick", labelsize=20)
matplotlib.rc("ytick", labelsize=20)

## Listes pour stocker les résultats
train_accuracies = [0.5]
test_accuracies =  [0.5]

# Itération sur quelques valeurs de profondeurs
for depth in range(1, 25):
    clf = tree.DecisionTreeClassifier(max_depth=depth)

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

    # Fit
    clf.fit(df_train[cols], df_train.quality)

    # training & test prediction
    train_predictions = clf.predict(df_train[cols])
    test_predictions = clf.predict(df_test[cols])

    # calcul de l'accuracy
    train_accuracy = metrics.accuracy_score(
        df_train.quality, train_predictions
    )
    test_accuracy = metrics.accuracy_score(
        df_test.quality, test_predictions
    )

    # ajout des résultats
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Affichage des résultats
plt.figure(figsize=(10, 5))
sns.set_style("whitegrid")
plt.plot(train_accuracies, label="train accuracy", linestyle="--")
plt.plot(test_accuracies, label="test accuracy")
plt.legend(loc="upper left", prop={'size': 15})
plt.xticks(range(0, 26, 5))
plt.xlabel("max_depth", size=20)
plt.ylabel("accuracy", size=20)
plt.show()

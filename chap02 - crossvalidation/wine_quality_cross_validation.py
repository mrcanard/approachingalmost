import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# affichage répartition
b = sns.countplot(x="quality", data=df)
b.set_xlabel("quality", fontsize=20)
b.set_ylabel("count", fontsize=20)
plt.show()

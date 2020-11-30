
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# トレーニングデータの取得
train = pd.read_csv("./titanic/train.csv")

# データ加工・前処理
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")
train = train.replace({"male": 0, "female": 1})
train = train.replace({"S": 0, "C": 1, "Q": 2})

target = train["Survived"].values
train_two = train.copy()
train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] + 1
features = train_two[["Pclass", "Age", "Sex",
                      "Fare", "family_size", "Embarked"]].values

# 決定木で学習
forest = GradientBoostingClassifier(n_estimators=55, random_state=9)
forest = forest.fit(features, target)

# 学習済みファイルの名前を設定
file = 'trained_model.pkl'
# Pickleファイルとして出力
pickle.dump(forest, open(file, 'wb'))

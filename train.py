
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv("./titanic/train.csv")

# データ加工
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")
train = train.replace({"male":0, "female":1})
train = train.replace({"S": 0, "C": 1, "Q": 2})

target = train["Survived"].values
train_two = train.copy()
train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] + 1
features = train_two[["Pclass", "Age", "Sex", "Fare", "family_size", "Embarked"]].values

forest = GradientBoostingClassifier(n_estimators=55, random_state=9)
forest = forest.fit(features, target)

file = 'trained_model.pkl'
pickle.dump(forest, open(file, 'wb'))

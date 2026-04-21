import numpy as np 
import pandas as pd 
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

training = pd.read_csv('/kaggle/input/competitions/titanic/train.csv')
testing = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')

y = training.Survived
features = ['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']
X = training[features]

X_test = testing[features]

model = HistGradientBoostingClassifier(categorical_features=features)

model.fit(X, y)

print(model.predict(X_test))

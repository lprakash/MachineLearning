import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Preprocessing/data1.csv')

from sklearn.preprocessing import Imputer
X = dataset.iloc[:, 0:3]
Y = dataset.iloc[:, 3]
imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
imp.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3] = imp.transform(X.iloc[:, 1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
X.iloc[:, 0] = le.fit_transform(X.iloc[:, 0])
Y = le.fit_transform(Y)

ohe = OneHotEncoder(categorical_features=[0])
X = ohe.fit_transform(X).toarray()
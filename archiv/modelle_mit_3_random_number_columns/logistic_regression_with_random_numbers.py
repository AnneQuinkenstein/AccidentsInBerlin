import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data_with_random_col = pd.read_csv('../data/GeneralDatensatz18-21ohneGeo-mitLockdown.csv', sep=';')

print(data_with_random_col.columns)

data_with_random_col['randNumCol'] = np.random.randint(0,1000, size=len(data_with_random_col))
data_with_random_col['randNumCol2'] = np.random.randint(0,1000, size=len(data_with_random_col))
data_with_random_col['randNumCol3'] = np.random.randint(0,1000, size=len(data_with_random_col))

X = data_with_random_col.drop('UKATEGORIE', axis=1)
X = X.drop('LOR_ab_2021', axis=1)
y = data_with_random_col['UKATEGORIE']

X_train, X_test, y_train, y_test = train_test_split(X, y)

lrc = LogisticRegression(max_iter=5000)
lrc.fit(X_train, y_train)
y_pred = lrc.predict(X_test)
print(y_pred)
print(y_test)
print(lrc.score(X_test, y_test))


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data_with_random_col = pd.read_csv('../data/GeneralDatensatz18-21ohneGeo-mitLockdown.csv', sep=';')
data_with_random_col['randNumCol'] = np.random.randint(0,1000, size=len(data_with_random_col))
data_with_random_col['randNumCol2'] = np.random.randint(0,1000, size=len(data_with_random_col))
data_with_random_col['randNumCol3'] = np.random.randint(0,1000, size=len(data_with_random_col))

X = data_with_random_col.drop('UKATEGORIE', axis=1)
y = data_with_random_col['UKATEGORIE']
X = X.drop('LOR_ab_2021', axis = 1)
# print(X.isnull().any())

X_train, X_test, y_train, y_test = train_test_split(X, y)
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)
print(score)

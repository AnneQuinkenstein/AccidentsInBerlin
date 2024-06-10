import os
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

csv_directory = '../data'
csv_file = '../data/GeneralDatensatz18-21ohneGeo.csv'
csv_path = os.path.join(csv_directory, csv_file)

df = pd.read_csv(csv_path, sep=';')

print(df.head())
print(df.describe())

# Wo fehlen Werte? Welche Zeilen?
missing_values = df.isnull().sum()

print("Fehlende Werte pro Spalte:")
print(missing_values)

missing_mask = df.isnull()

print("Positionen der fehlenden Werte:")
for col in missing_mask.columns:
    if missing_mask[col].any():
        missing_rows = missing_mask[missing_mask[col]].index.tolist()
        print(f"Spalte '{col}' hat fehlende Werte in den Zeilen: {missing_rows}")

# LOR Daten missing: Jahr 2021 obj 214125, 2020 obj 136842, 2018 obj 199484, 2018 obj 200009, 2018 obj 200322 missing

y=df['UKATEGORIE']
X=df[['UMONAT','USTUNDE','UWOCHENTAG','UART','USTRZUSTAND','BEZ','UTYP1','ULICHTVERH','IstRad','IstPKW','IstFuss','IstKrad','IstGkfz','IstSonstige']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



logR = LogisticRegression(max_iter = 5000)
logR.fit(X_train, y_train)

y_pred = logR.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logR.score(X_test, y_test)))

clf = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

plt.figure(figsize=(30,15))

plot_tree(clf,filled=True, feature_names=X.columns)
plt.savefig('treeAcc.png')

plt.show()

# Wichtigkeit der Merkmale anzeigen
feature_importances = pd.DataFrame(clf.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
print("Wichtigkeit der Merkmale:")
print(feature_importances)
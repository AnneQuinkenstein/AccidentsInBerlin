from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

file_path = 'AfSBBB_BE_LOR_Strasse_Strassenverkehrsunfaelle_2021_Datensatz.csv'
df = pd.read_csv(file_path, sep=';')

print(df.head())
print(df.describe())


y=df['UART']
X=df[['UMONAT','USTUNDE','UWOCHENTAG','UKATEGORIE','USTRZUSTAND','BEZ','UTYP1','ULICHTVERH','IstRad','IstPKW','IstFuss','IstKrad','IstGkfz','IstSonstige']]

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
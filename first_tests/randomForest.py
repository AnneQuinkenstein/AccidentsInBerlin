import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Laden der Daten
data = pd.read_csv('../data/GeneralDatensatz18-21ohneGeo.csv', sep=';')

duplicate_objectids = data[data.duplicated('OBJECTID', keep=False)]

# Anzahl der doppelten OBJECTID-Werte
num_duplicates = duplicate_objectids.shape[0]

if num_duplicates > 0:
    print(f'Es gibt {num_duplicates} doppelte OBJECTID-Werte:')
    print(duplicate_objectids)
else:
    print('Es gibt keine doppelten OBJECTID-Werte.')

# Vorbereiten der Daten
X = data.drop('UKATEGORIE', axis=1)
y = data['UKATEGORIE']

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Wichtigkeit der Merkmale
rf_feature_importances = pd.DataFrame(rf_clf.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
print("Wichtigkeit der Merkmale (Random Forest):")
print(rf_feature_importances)

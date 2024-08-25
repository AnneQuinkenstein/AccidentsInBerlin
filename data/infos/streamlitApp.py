import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Laden der Daten
data = pd.read_csv('../data/GeneralDatensatz18-21ohneGeo.csv', sep=';')

# Titel der App
st.title("Straßenverkehrsunfälle in Berlin")

# Zeige die ersten paar Zeilen des DataFrames an
st.write("Erste fünf Zeilen der Daten:")
st.write(data.head())

liste_feature_drop= ['LOR_ab_2021', 'UKATEGORIE']

# Vorbereiten der Daten
X = data.drop(liste_feature_drop, axis=1)
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

# Zeige die Wichtigkeit der Merkmale in der Streamlit-App an
st.write("Wichtigkeit der Merkmale (Random Forest):")
st.write(rf_feature_importances)



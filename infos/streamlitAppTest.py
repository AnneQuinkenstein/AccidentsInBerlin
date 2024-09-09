
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Daten laden und Modell laden
data = pd.read_csv('../data/GeneralDatensatz18-21ohneGeo.csv', sep=';')
#model_filename = "../modelle/rf_featureImportanzen.pkl" # Pfad zum Modell schwere und tödliche unfälle vorherzusagen

# Modell laden
#with open(model_filename, 'rb') as file:
#    model = pickle.load(file)

# Benutzeroberfläche
st.title("Unfallschwere Vorhersage")
st.sidebar.header("Eingabedaten")

#Bundesland
land_mapping = {
    "Berlin": 11
}

#wochentage
weekday_mapping = {
    "Montag": 2,
    "Dienstag": 3,
    "Mittwoch": 4,
    "Donnerstag": 5,
    "Freitag": 6,
    "Samstag": 7,
    "Sonntag": 1
}

#Monat
month_mapping = {
    "Januar": 1,
    "Februar": 2,
    "März": 3,
    "April": 4,
    "Mai": 5,
    "Juni": 6,
    "Juli": 7,
    "August": 8,
    "September": 9,
    "Oktober": 10,
    "November": 11,
    "Dezember": 12
}

#Bezirkmapping
district_mapping = {
    1: "Mitte ",
    2: "Friedrichshain-Kreuzberg ",
    3: "Pankow ",
    4: "Charlottenburg-Wilmersdorf ",
    5: "Spandau ",
    6: "Steglitz-Zehlendorf ",
    7: "Tempelhof-Schöneberg ",
    8: "Neukölln ",
    9: "Treptow-Köpenick ",
    10: "Marzahn-Hellersdorf ",
    11: "Lichtenberg ",
    12: "Reinickendorf "
}
# Convert the district numbers in the data to the formatted names
data['BEZ'] = data['BEZ'].map(district_mapping)
reverse_district_mapping = {v.strip(): k for k, v in district_mapping.items()}


# Straßenzustandszuordnung
road_condition_mapping = {
    0: "trocken",
    1: "nass/feucht/schlüpfrig",
    2: "winterglatt"
}
# Convert the road condition numbers in the data to the descriptions
data['USTRZUSTAND'] = data['USTRZUSTAND'].map(road_condition_mapping)
reverse_road_condition_mapping = {v: k for k, v in road_condition_mapping.items()}

# Lichtverhältnisse
light_condition_mapping = {
    "Tageslicht": 0,
    "Dämmerung": 1,
    "Dunkelheit": 2
}

# Eingabefelder in der Sidebar
land_name = st.sidebar.selectbox("Bundesland", list(land_mapping.keys()), index=0)
land = land_mapping[land_name]
hour = st.sidebar.slider("Stunde", 0, 23, 0)
weekday_name = st.sidebar.selectbox("Wochentag", list(weekday_mapping.keys()), index=0)
weekday = weekday_mapping[weekday_name] # Mapping des Wochentags auf die numerische Darstellung
month_name = st.sidebar.selectbox("Monat", list(month_mapping.keys()), index=0)
month = month_mapping[month_name]
district_name = st.sidebar.selectbox("Bezirk", list(reverse_district_mapping.keys()))
district = reverse_district_mapping[district_name]
road_condition_name = st.sidebar.selectbox("Straßenzustand", list(reverse_road_condition_mapping.keys()))
road_condition = reverse_road_condition_mapping[road_condition_name]
light_condition_name = st.sidebar.selectbox("Lichtverhältnisse", list(light_condition_mapping.keys()))
light_condition = light_condition_mapping[light_condition_name]

vehicle_type = st.sidebar.radio("Verkehrsmittel", ['Rad', 'Auto', 'Kraftrad', 'Zu Fuß', 'Lastkraftwagen', 'Sonstige'])

# Abbildung der Eingabedaten auf die Merkmale des Modells
input_data = {
    "LAND": land,
    "BEZ": district,
    #"LOR_ab_2021": 0,
    "UJAHR": 0,
    "UMONAT": month,
    "USTUNDE": hour,
    "UWOCHENTAG": weekday,
    #"UKATEGORIE": 0,
    "UART": 0,
    "UTYP1": 0,
    "ULICHTVERH": light_condition,
    "IstRad": 1 if vehicle_type == 'Rad' else 0,
    "IstPKW": 1 if vehicle_type == 'Auto' else 0,
    "IstFuss": 1 if vehicle_type == 'Zu Fuß' else 0,
    "IstKrad": 1 if vehicle_type == 'Kraftrad' else 0,
    "IstGkfz": 1 if vehicle_type == 'Lastkraftwagen' else 0,
    "IstSonstige": 1 if vehicle_type == 'Sonstige' else 0,
    "USTRZUSTAND": road_condition
  }

# Vorhersage
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Vorhersage")
st.write(f"Unfallkategorie: {prediction[0]}")

st.subheader("Wahrscheinlichkeit")
st.bar_chart(prediction_proba[0])



# Extrahieren der Feature-Importanzen
feature_importances = model.feature_importances_
features = ["LAND", "BEZ", "UJAHR", "UMONAT", "USTUNDE", "UWOCHENTAG", "UART", "UTYP1", "ULICHTVERH", "IstRad", "IstPKW", "IstFuss", "IstKrad", "IstGkfz", "IstSonstige", "USTRZUSTAND"]

# Erstellen eines DataFrames für die Feature-Importanzen
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Anzeige der Feature-Importanzen in der Streamlit-App
st.subheader("Feature-Importanzen für die Unfallschwere (UKATEGORIE 2)")
st.bar_chart(importance_df.set_index('Feature'))

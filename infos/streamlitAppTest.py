import joblib
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Daten laden und Modell laden
data = pd.read_csv('../data/GeneralDatensatz18-21ohneGeo.csv', sep=';')
model_filename = "../modelle_getuned/random_forest_model.pkl" # Pfad zum Modell schwere und tödliche unfälle vorherzusagen
model = joblib.load(model_filename)

# Benutzeroberfläche
st.title("Wird ein Unfall in Berlin leicht oder schwer*?")
st.subheader("Bitte gib links die Bedingungen ein, unter denen du einen Unfall befürchtest. Mit den open Data von Berlin wird die Wahrscheinlichkeit für die Unfallschwere mit dem Machine Learning-Algorithmus 'Random Forest' vorhergesagt.")
st.write("*Wir können keine Aussage darüber treffen, ob du einen Unfall haben wirst und können keine Aussage über die Unfallwahrscheinlichkeit geben.")
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
ferien = st.sidebar.selectbox("Ferien", ["Ja", "Nein"])
covid = st.sidebar.selectbox("COVID", ["Nein", "Ja"])

# Abbildung der Eingabedaten auf die Merkmale des Modells
input_data = {
    "BEZ": [district],
    "UJAHR": [0],
    "UMONAT": [month],
    "USTUNDE": [hour],
    "UWOCHENTAG": [weekday],
    "UART": [0],
    "UTYP1": [0],
    "ULICHTVERH": [light_condition],
    "IstRad": [1 if vehicle_type == 'Rad' else 0],
    "IstPKW": [1 if vehicle_type == 'Auto' else 0],
    "IstFuss": [1 if vehicle_type == 'Zu Fuß' else 0],
    "IstKrad": [1 if vehicle_type == 'Kraftrad' else 0],
    "IstGkfz": [1 if vehicle_type == 'Lastkraftwagen' else 0],
    "IstSonstige": [1 if vehicle_type == 'Sonstige' else 0],
    "USTRZUSTAND": [road_condition],
    "LOCKDOWN": [0],
    "COVID": [1 if covid == "Ja" else 0],
    "FERIEN": [1 if ferien == "Ja" else 0]
}

# Vorhersage
input_df = pd.DataFrame(input_data)
prediction = model.predict(input_df)[0] # [0] is used to get the prediction as a scalar instead of an array
prediction_proba = model.predict_proba(input_df)[0]

# Ausgabe der Ergebnisse
st.subheader("Wahrscheinlichkeitsverteilung für Schwere eines Unfalls")
categories = {1: "Leicht", 0: "Schwer"}

# Wahrscheinlichkeiten anzeigen
probabilities = {
    "Leicht": prediction_proba[1],
    "Schwer": prediction_proba[0]
}
# Erstellen eines DataFrames für das Balkendiagramm
probabilities_df = pd.DataFrame(probabilities, index=["Wahrscheinlichkeit"])

# Balkendiagramm anzeigen
st.bar_chart(probabilities_df.T)

# Wahrscheinlichkeiten in Prozent umrechnen
leicht_prozent = probabilities["Leicht"] * 100
schwer_prozent = probabilities["Schwer"] * 100

# Zusammenfassende Anzeige der Wahrscheinlichkeiten
st.write(f"Falls du unter den von dir eingegebenen Bedingungen einen Unfall hast, beträgt die Wahrscheinlichkeit dafür, dass der Unfall leicht ausfällt {leicht_prozent:.2f}% und dafür dass er schwer ausfällt {schwer_prozent:.2f}%. Es wird somit erwartet, dass der Unfall wenn dann eher {categories[prediction]} ist. Angaben ohne Gewähr.")

# Bild anzeigen
st.subheader("Ranking der Einflussfaktoren auf die Vorhersage")
st.image("WichtigkeitFeatureRanking.png", caption="Wichtigkeit der Features im Modell")

# Hinzufügen der Map
st.subheader("Unfallpunkte in Berlin 2018-2021")
st.image("../infos/mapAccidentPoints.png", caption="Unfallpunkte auf der Berlinkarte")

st.write("Open Data Portal Source: https://daten.berlin.de/datensaetze?q=strassenverkehrsunfalle%2520nach%2520Unfallort&sort=score+desc%2C+metadata_modified+desc")
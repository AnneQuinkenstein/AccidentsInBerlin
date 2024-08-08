import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiLineString, LineString
from rtree import index
import time

csv_file_path = '../data/GeneralDatensatz18-21.csv'
geojson_file_path = '../data/filtered_osm_highway_v1.geojson'

# Accident
start_time = time.time()
df = pd.read_csv(csv_file_path, delimiter=';')
csv_load_time = time.time() - start_time
print(f"Zeit zum Laden der CSV-Datei: {csv_load_time:.2f} Sekunden")

# XGCSWGS84` und `YGCSWGS84` als floats anstatt von strings mit Kommas
if df['XGCSWGS84'].dtype == object:
    df['XGCSWGS84'] = df['XGCSWGS84'].str.replace(',', '.').astype(float)
if df['YGCSWGS84'].dtype == object:
    df['YGCSWGS84'] = df['YGCSWGS84'].str.replace(',', '.').astype(float)

# highway residential, primary, secondary, tertiary, service, living_street
start_time = time.time()
geo_df = gpd.read_file(geojson_file_path)
geojson_load_time = time.time() - start_time
print(f"Zeit zum Laden der GeoJSON-Datei: {geojson_load_time:.2f} Sekunden")

# MultiLineString zu LineString, damit Punkte auf Linien liegen
start_time = time.time()
geo_df['line_strings'] = geo_df['geometry'].apply(lambda geom: list(geom.geoms) if isinstance(geom, MultiLineString) else [geom])
line_strings = [line for lines in geo_df['line_strings'] for line in lines]
geometry_processing_time = time.time() - start_time
print(f"Zeit für die Umwandlung der Geometrien: {geometry_processing_time:.2f} Sekunden")

# Erstellen eines räumlichen Indexes für die LineStrings und begrenzung der linestrings in den index eingefügt
spatial_index = index.Index()
for pos, line in enumerate(line_strings):
    spatial_index.insert(pos, line.bounds)

# räuml. Index nutzen, um die Anwärter rauszufiltern, die in der Nähe eines Punkts sind, dann prüfen ob innerhalb eines gepufferten LineStrings
def get_linestring_index_if_contains_point(point, line_strings, spatial_index):
    # Kandidaten aus dem räumlichen Index abrufen
    candidate_idxs = list(spatial_index.intersection(point.bounds))
    for idx in candidate_idxs:
        line = line_strings[idx]
        if point.within(line.buffer(0.0001)):  # Verwende einen kleinen Puffer, falls die Punkte auf den Linien liegen sollen
            return idx
    return None

start_time = time.time()
result = []
highway_categories = []
for idx, row in df.iterrows():
    point = Point(row['XGCSWGS84'], row['YGCSWGS84'])
    line_idx = get_linestring_index_if_contains_point(point, line_strings, spatial_index)
    in_cycle_net = line_idx is not None
    result.append({'XGCSWGS84': row['XGCSWGS84'], 'YGCSWGS84': row['YGCSWGS84'], 'in_cycle_net': in_cycle_net})
    if line_idx is not None:
        highway_categories.append(geo_df.iloc[line_idx]['highway'])
    else:
        highway_categories.append(None)

coordinate_check_time = time.time() - start_time
print(f"Zeit für die Überprüfung der Koordinaten: {coordinate_check_time:.2f} Sekunden")

# Ergebnis als DataFrame
result_df = pd.DataFrame(result)

# Hinzufügen der highway Spalte zu df
df['highway'] = highway_categories

# Speichern der aktualisierten CSV-Datei
df.to_csv('../data/GeneralDatensatz18-21_with_highway.csv', index=False)

# Ausgabe der ersten Zeilen des Ergebnisses
print(result_df.head())
print(result_df.describe())
print(result_df['in_cycle_net'].value_counts())
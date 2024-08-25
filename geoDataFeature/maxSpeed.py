import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiLineString, LineString
from rtree import index
import time


csv_file_path = '../data/GeneralDatensatz18-21.csv'
geojson_file_path = '../data/cycle_net_berlin_cleaned_maxspeed.geojson'

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

# maxSpeed 20/30/50/60 kmh
start_time = time.time()
geo_df = gpd.read_file(geojson_file_path)
geojson_load_time = time.time() - start_time
print(f"Zeit zum Laden der GeoJSON-Datei: {geojson_load_time:.2f} Sekunden")

#Unfallpunkte als auch die Linien im selben Koordinatensystem (CRS)? Transform Geodaten einheitliches CRS
df_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.XGCSWGS84, df.YGCSWGS84), crs="EPSG:4326")
if geo_df.crs != df_gdf.crs:
    geo_df = geo_df.to_crs(df_gdf.crs)


# MultiLineString zu LineString, damit Punkte auf Linien liegen
start_time = time.time()
geo_df['line_strings'] = geo_df['geometry'].apply(lambda geom: list(geom.geoms) if isinstance(geom, MultiLineString) else [geom])
line_strings = [line for lines in geo_df['line_strings'] for line in lines]
geometry_processing_time = time.time() - start_time
print(f"Zeit für die Umwandlung der Geometrien: {geometry_processing_time:.2f} Sekunden")

#Problem Ladezeit weil data zu groß
# Erstellen eines räumlichen Indexes für die LineStrings und begrenzung der linestrings in den index eingefügt
spatial_index = index.Index()
for pos, line in enumerate(line_strings):
    spatial_index.insert(pos, line.bounds)

# räuml. Index nutzen, um die Anwärter rauszufiltern, die in der Nähe eines Punkts sind, dann prüfen ob innerhalb eines gepufferten LineStrings
def is_point_in_linestrings(point, line_strings, spatial_index, buffer_distance=0.0005):
    # Kandidaten aus dem räumlichen Index abrufen
    candidate_idxs = list(spatial_index.intersection(point.bounds))
    for idx in candidate_idxs:
        line = line_strings[idx]
        if point.within(line.buffer(buffer_distance)):
            return True
    return False

buffer_distances = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.1]  # Verschiedene Puffergrößen
for distance in buffer_distances:
    result = []
    for idx, row in df.iterrows():
        point = Point(row['XGCSWGS84'], row['YGCSWGS84'])
        in_cycle_net = is_point_in_linestrings(point, line_strings, spatial_index, buffer_distance=distance)
        result.append({'XGCSWGS84': row['XGCSWGS84'], 'YGCSWGS84': row['YGCSWGS84'], 'in_cycle_net': in_cycle_net})
    result_df = pd.DataFrame(result)
    print(f"Für Puffergröße {distance}:")
    print(result_df['in_cycle_net'].value_counts())


start_time = time.time()
result = []
for idx, row in df.iterrows():
    point = Point(row['XGCSWGS84'], row['YGCSWGS84'])
    in_cycle_net = is_point_in_linestrings(point, line_strings, spatial_index, buffer_distance=0.0005)
    result.append({'XGCSWGS84': row['XGCSWGS84'], 'YGCSWGS84': row['YGCSWGS84'], 'in_cycle_net': in_cycle_net})
coordinate_check_time = time.time() - start_time
print(f"Zeit für die Überprüfung der Koordinaten: {coordinate_check_time:.2f} Sekunden")

# Ergebnis als DataFrame
result_df = pd.DataFrame(result)

# Ausgabe der ersten Zeilen des Ergebnisses
print(result_df.head())
print(result_df.describe())
print(result_df['in_cycle_net'].value_counts())
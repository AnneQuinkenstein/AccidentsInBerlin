import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiLineString, LineString
from rtree import index
import time


csv_file_path = '../data/GeneralDatensatz18-21.csv'
geojson_file_path_highway = '../data/filtered_osm_highway_v1.geojson'
geojson_file_path_speed = '../data/cycle_net_berlin_cleaned_maxspeed.geojson'
geojson_file_path_surface = '../data/cycle_net_berlin_cleaned_surface.geojson'

# Accident
start_time = time.time()
df = pd.read_csv(csv_file_path, delimiter=';')
csv_load_time = time.time() - start_time
print(f"Zeit zum Laden der CSV-Datei: {csv_load_time:.2f} Sekunden")

# XGCSWGS84 und YGCSWGS84 als floats verarbeiten
if df['XGCSWGS84'].dtype == object:
    df['XGCSWGS84'] = df['XGCSWGS84'].str.replace(',', '.').astype(float)
if df['YGCSWGS84'].dtype == object:
    df['YGCSWGS84'] = df['YGCSWGS84'].str.replace(',', '.').astype(float)

# highway residential, primary, secondary, tertiary, service, living_street
# surface asphalt, unpaved, concrete
# maxSpeed 5/20/30/50/60 kmh
start_time = time.time()
geo_df_highway = gpd.read_file(geojson_file_path_highway)
geo_df_speed = gpd.read_file(geojson_file_path_speed)
geo_df_surface = gpd.read_file(geojson_file_path_surface)
geojson_load_time = time.time() - start_time

# MultiLineString zu LineString, damit Punkte auf Linien liegen
def convert_multilinestrings_to_linestrings(geo_df):
    geo_df['line_strings'] = geo_df['geometry'].apply(lambda geom: list(geom.geoms) if isinstance(geom, MultiLineString) else [geom])
    return [line for lines in geo_df['line_strings'] for line in lines]
start_time = time.time()
line_strings_highway = convert_multilinestrings_to_linestrings(geo_df_highway)
line_strings_speed = convert_multilinestrings_to_linestrings(geo_df_speed)
line_strings_surface = convert_multilinestrings_to_linestrings(geo_df_surface)
geometry_processing_time = time.time() - start_time
print(f"Zeit für die Umwandlung der Geometrien: {geometry_processing_time:.2f} Sekunden")

#Problem Ladezeit weil data zu groß
# Erstellen eines räumlichen Indexes für die LineStrings und begrenzung der linestrings in den index eingefügt
def create_spatial_index(line_strings):
    spatial_index = index.Index()
    for pos, line in enumerate(line_strings):
        spatial_index.insert(pos, line.bounds)
    return spatial_index

spatial_index_highway = create_spatial_index(line_strings_highway)
spatial_index_speed = create_spatial_index(line_strings_speed)
spatial_index_surface = create_spatial_index(line_strings_surface)
print("Spatial Index created")
# Funktion zur Abfrage der line_strings Indizes basierend auf den Punkten
def get_linestring_index_if_contains_point(point, line_strings, spatial_index):
    candidate_idxs = list(spatial_index.intersection(point.bounds))
    for idx in candidate_idxs:
        if idx < len(line_strings):  # Ensure idx is within bounds
            line = line_strings[idx]
            if point.within(line.buffer(0.0001)):
                return idx
    return None

# Koordinatenüberprüfung und Zuordnung von highway, maxspeed_category und surface_category
start_time = time.time()
highway_categories = []
maxSpeed_categories = []
surface_categories = []
for idx, row in df.iterrows():
    point = Point(row['XGCSWGS84'], row['YGCSWGS84'])

    line_idx_highway = get_linestring_index_if_contains_point(point, line_strings_highway, spatial_index_highway)
    if line_idx_highway is not None and line_idx_highway < len(geo_df_highway):
        highway_categories.append(geo_df_highway.iloc[line_idx_highway]['highway'])
    else:
        highway_categories.append(None)


    line_idx_speed = get_linestring_index_if_contains_point(point, line_strings_speed, spatial_index_speed)
    if line_idx_speed is not None and line_idx_speed < len(geo_df_speed):
        maxSpeed_categories.append(geo_df_speed.iloc[line_idx_speed]['maxspeed_category'])
    else:
        maxSpeed_categories.append(None)

    line_idx_surface = get_linestring_index_if_contains_point(point, line_strings_surface, spatial_index_surface)
    if line_idx_surface is not None and line_idx_surface < len(geo_df_surface):
        surface_categories.append(geo_df_surface.iloc[line_idx_surface]['surface_category'])
    else:
        surface_categories.append(None)


coordinate_check_time = time.time() - start_time
print(f"Zeit für die Überprüfung der Koordinaten: {coordinate_check_time:.2f} Sekunden")

# Hinzufügen der neuen Spalten zu df
df['highway'] = highway_categories
df['maxspeed_category'] = maxSpeed_categories
df['surface_category'] = surface_categories

# Speichern der aktualisierten CSV-Datei
df.to_csv('../data/GeneralDatensatz18-21_with_additional_info.csv', index=False)

# Ausgabe der ersten Zeilen des Ergebnisses
print(df.head())
print(df.describe())
print(df[['highway', 'maxspeed_category', 'surface_category']].value_counts())

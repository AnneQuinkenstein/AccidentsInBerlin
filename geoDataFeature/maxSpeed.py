import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiLineString, mapping, shape, LineString

#Pfade zu den Dateien
csv_file_path = '../data/GeneralDatensatz18-21.csv'
geojson_file_path = '../data/cycle_net_berlin_cleaned_maxspeed.geojson'

# Laden der CSV-Datei
df = pd.read_csv(csv_file_path, delimiter=';')

# Überprüfen, ob die Spalten XGCSWGS84 und YGCSWGS84 als Strings eingelesen werden
if df['XGCSWGS84'].dtype == object:
    df['XGCSWGS84'] = df['XGCSWGS84'].str.replace(',', '.').astype(float)
if df['YGCSWGS84'].dtype == object:
    df['YGCSWGS84'] = df['YGCSWGS84'].str.replace(',', '.').astype(float)

# Laden des GeoJSON-Files
geo_df = gpd.read_file(geojson_file_path)

# Umwandlung der MultiLineString-Objekte in einzelne LineString-Objekte
line_strings = []
for geom in geo_df['geometry']:
    if isinstance(geom, MultiLineString):
        line_strings.extend([line for line in geom.geoms])
    elif isinstance(geom, LineString):
        line_strings.append(geom)

# Funktion zur Überprüfung, ob ein Punkt in einem der LineStrings enthalten ist
def is_point_in_linestrings(point, line_strings):
    for line in line_strings:
        if point.within(line.buffer(0.0001)):  # Verwende einen kleinen Puffer, falls die Punkte auf den Linien liegen sollen
            return True
    return False

# Überprüfung der Koordinaten in der CSV-Datei
result = []
for idx, row in df.iterrows():
    point = Point(row['XGCSWGS84'], row['YGCSWGS84'])
    in_cycle_net = is_point_in_linestrings(point, line_strings)
    result.append({'XGCSWGS84': row['XGCSWGS84'], 'YGCSWGS84': row['YGCSWGS84'], 'in_cycle_net': in_cycle_net})

# Ergebnis als DataFrame
result_df = pd.DataFrame(result)

# Ausgabe der ersten Zeilen des Ergebnisses
print(result_df.head())
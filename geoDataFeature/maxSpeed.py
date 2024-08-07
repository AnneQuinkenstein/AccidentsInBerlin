import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape

# Pfade zu den Dateien
csv_file_path = '../data/GeneralDatensatz18-21.csv'
geojson_file_path = '../data/cycle_net_berlin_cleaned_maxspeed.geojson'

# Laden der CSV-Datei
df = pd.read_csv(csv_file_path)

# Laden des GeoJSON-Files
geo_df = gpd.read_file(geojson_file_path)

# Konvertieren der GeoJSON-Geometrien in Shapely-Objekte
geometries = [shape(feature['geometry']) for feature in geo_df['geometry']]

# Funktion, um zu überprüfen, ob ein Punkt in einer der Geometrien liegt
def point_in_geometries(x, y, geometries):
    point = Point(x, y)
    return any(geometry.contains(point) for geometry in geometries)

# Überprüfen für jedes Objekt in der CSV-Datei
df['in_geojson'] = df.apply(lambda row: point_in_geometries(row['XGCSWGS84'], row['YGCSWGS84'], geometries), axis=1)

# Ergebnis anzeigen
print(df.head())

# Optional: Speichern der Ergebnisse in eine neue CSV-Datei
df.to_csv('path/to/GeneralDatensatz18-21_checked.csv', index=False)

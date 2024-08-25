import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiLineString, LineString

# Load the accident points CSV
csv_file_path = '../data/GeneralDatensatz18-21.csv'
df = pd.read_csv(csv_file_path, delimiter=';')

# Convert XGCSWGS84 and YGCSWGS84 to floats
if df['XGCSWGS84'].dtype == object:
    df['XGCSWGS84'] = df['XGCSWGS84'].str.replace(',', '.').astype(float)
if df['YGCSWGS84'].dtype == object:
    df['YGCSWGS84'] = df['YGCSWGS84'].str.replace(',', '.').astype(float)

# Create a GeoDataFrame for the accident points
df_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.XGCSWGS84, df.YGCSWGS84), crs="EPSG:4326")

# Load the line strings GeoJSON
geojson_file_path = '../data/cycle_net_berlin_cleaned_surface.geojson'
geo_df = gpd.read_file(geojson_file_path)

# Ensure both GeoDataFrames have the same CRS
if geo_df.crs != df_gdf.crs:
    geo_df = geo_df.to_crs(df_gdf.crs)

# Plot the accident points and line strings
fig, ax = plt.subplots(figsize=(30, 30))
geo_df.plot(ax=ax, color='blue', linewidth=1, label='Line Strings')
df_gdf.plot(ax=ax, color='red', markersize=5, label='Accident Points')
plt.legend()
plt.title('Accident Points and Line Strings')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
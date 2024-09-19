import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

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

# Plot the accident points
fig, ax = plt.subplots(figsize=(30, 30))
df_gdf.plot(ax=ax, color='red', markersize=5, label='Accident Points')
plt.legend()
plt.title('Accident Points')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.show()
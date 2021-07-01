# predictions_plot.py

import pandas as pd
import geopandas as gpd
from shapely import wkt

# return a plot of the predictions data visualized by col, which is a
# string identifying a column in the predictions_csv
def predictions_plot(predictions_csv, col):
    df = pd.read_csv(predictions_csv)
    df['geometry'] = df['geometry'].apply(wkt.loads) # fix geometry
    gdf = gpd.GeoDataFrame(df, crs='epsg:4326')
    plot = gdf.plot(column = col)
    return plot

# example usage
# import matplotlib.pyplot as plt
# predictions_plot('decade1_pred_filename.csv', 'prediction_proba')
# plt.show()


# preprocess_points.py
# spatially thin AIS point data

import ee
from src.gee_funs import filter_date_space, merge_coll

def spatially_thin(point_asset_path, start_year, end_year, distance=5000):
    points = ee.FeatureCollection(point_asset_path)
    study_dates = ee.List(list(map( \
            lambda x: ee.Date(str(x) + '-01-01'), \
            range(start_year, end_year))))
    feats = study_dates.map(lambda x: filter_date_space(x, points, distance))
    spatially_thin = ee.FeatureCollection(feats.iterate(merge_coll, points))
    export = ee.batch.Export.table.toAsset(collection = spatially_thin,
        description = 'sthin',
        assetId = point_asset_path + '_sthin')
    export.start()


#Import ee and required packages
import ee
ee.Initialize()
import pandas as pd 
import numpy as np
import glob
import geopandas as gpd


# Spatially thin locations and export to asset
#//========================================================

def filterDistance(points, distance):
    ## EE FUNCTION THAT TAKES A FEATURE COLLECTION AND FILTERS 
    ## BY A MINIMUM DISTANCE IN METERS
    def iter_func(el, ini):
        ini = ee.List(ini)
        fcini = ee.FeatureCollection(ini)
        buf = ee.Feature(el).geometry().buffer(distance)
        s = fcini.filterBounds(buf).size()
        cond = s.lte(0)
        return ee.Algorithms.If(cond, ini.add(el), ini)
    filt2 = ee.List([])
    filt = points.iterate(iter_func, filt2)
    filtered = ee.FeatureCollection(ee.List(filt))
    return filtered

#//========================================================




# Filter and spatially thin the feature collection
#//========================================================

def filter_date_space(date, collection, distance):
    start_date = ee.Date(date).advance(1, 'year')
    end_date = start_date.advance(1, 'year')
    points_in_that_year = collection.filterDate(start_date, end_date)

    spatially_filt = filterDistance(points_in_that_year, distance)

    return spatially_filt 

#//========================================================




# Does........ Something?
#//========================================================

def merge_coll(first_year, second_year):
    return ee.FeatureCollection(first_year).merge(ee.FeatureCollection(second_year))

#//========================================================



#embed observation Year as system:start_time for thinned dataset
#//========================================================

def embedd_date(x):
    yr = ee.Number(x.get("Year"))
    eedate = ee.Date.fromYMD(yr, 1, 1)
    return x.set("system:time_start", eedate)

#//========================================================





## Mask features by quality control bands
# ========================================================
# Masking TPP via quality control bands
# ========================================================
def gpp_qc(img):
    img2 = img.rename(['GPP','QC']);
    quality = img2.select("QC")
    mask = quality.neq(11) \
                .And(quality.neq(10)) \
                .And(quality.neq(20)) \
                .And(quality.neq(21)) 
    return img2.mask(mask)

#//========================================================



# ========================================================
# Masking LST via quality control bands
# ========================================================
def lst_qc(img):
    quality = img.select("QC_Day")
    mask = quality.bitwiseAnd(3).eq(0) \
                .And(quality.bitwiseAnd(12).eq(0))
    return img.mask(mask)

#//========================================================





#//========================================================
# ========================================================
# Mask Modus Vegetation Indices by quality flag
# ========================================================
def modusQC(image):
    quality = image.select("SummaryQA")
    mask = quality.eq(0)
    return image.updateMask(mask)

#//========================================================





#//========================================================
# ========================================================
# Mask Continuous Fields via quality control band
# ========================================================
def VCFqc(img):
    quality = img.select("Quality")
    mask = quality.bitwiseAnd(2).eq(0) \
                    .And(quality.bitwiseAnd(4).eq(0)) \
                    .And(quality.bitwiseAnd(8).eq(0)) \
                    .And(quality.bitwiseAnd(16).eq(0)) \
                    .And(quality.bitwiseAnd(32).eq(0))

    return img.mask(quality)

#//========================================================


#//========================================================
#========================================================
# REDUCE REGIONS FUNCTION
#========================================================
def reduce_HUCS(img,thinned_map, HUC_clip):
    # Cast to image because EE finicky reasons
    img = ee.Image(img)
    startDate = ee.Date(img.get("system:time_start"))
    endDate = startDate.advance(1,'year')
    
    # Find each occurrence record within the year
    pointsInThatYear = thinned_map.filterDate(startDate, endDate);
   
    # Define Point Joins such that each HUC contains a list of observational data:
    distFilter = ee.Filter.intersects(**{
      'leftField': '.geo',
      'rightField': '.geo',
      'maxError': 100
    });
    
    pointJoin = ee.Join.saveAll(**{
        'matchesKey': 'Points',
    });()
    
    # Apply spatial join to presence data 
    new_HUCS = pointJoin.apply(HUC_clip,pointsInThatYear,distFilter);
    
    #Reduce the Multi-band image to HUC-level aggregates
    reduced_image = img.reduceRegions(**{
                              'collection': new_HUCS,
                              'reducer': ee.Reducer.mean(),
                              'crs': 'EPSG:4326',
                              'scale': 100,
                              'tileScale': 16}).map(lambda y:y.set({'Time': img.get("system:time_start")}))
    
    def getAvgPresence (feat):
        pts = ee.List(feat
              .get('Points')) \
              .map(lambda pt: ee.Feature(pt).get('Present'))
                          
        avg = ee.List(pts).reduce(ee.Reducer.mean())
   
        return feat.set("Avg_Presence", avg)
    
    return reduced_image.map(getAvgPresence)

#//========================================================









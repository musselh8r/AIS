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





##//========================================================
### Annual Cube function
##========================================================
## "Builder Function" -- processes each annual variable into a list of images
##========================================================
#
#def build_annual_cube(d):
#    # Set start and end dates for filtering time dependent predictors (SR, NDVI, Phenology)
#      # Advance startDate by 1 to begin with to account for water year (below)
#    startDate = (ee.Date(d).advance(1.0,'year').millis()) ## FIXME: Why do we advance a year? this give 2003-2019 instead of 2002-2018
#    endDate = ee.Date(d).advance(2.0,'year').millis()
#
#  #========================================================
#  #Define function to compute seasonal information for a given variable
#  #========================================================
#    def add_seasonal_info(imgCol,name,bandName):
#        winter = imgCol.filterDate(winter_start,winter_end)
#        spring = imgCol.filterDate(spring_start,spring_end)
#        summer = imgCol.filterDate(summer_start,summer_end)
#        fall = imgCol.filterDate(fall_start,fall_end)
#
#        winter_tot = winter.sum()
#        spring_tot = spring.sum()
#        summer_tot = summer.sum()
#        fall_tot = fall.sum()
#
#        winter_max = winter.max()
#        winter_min = winter.min()
#        spring_max = spring.max()
#        spring_min = spring.min()
#        summer_max = summer.max()
#        summer_min = summer.min()
#        fall_max = fall.max()
#        fall_min = fall.min()
#
#        winter_diff = winter_max.subtract(winter_min)
#        spring_diff = spring_max.subtract(spring_min)
#        summer_diff = summer_max.subtract(summer_min)
#        fall_diff = fall_max.subtract(fall_min)
#
#        names = ['winter_total'+name,'spring_total'+name,'summer_total'+name,
#                      'fall_total'+name]
#
#        return winter_tot.addBands([spring_tot,summer_tot,fall_tot]) \
#                         .rename(names)
#
#  # Set up Seasonal dates for precip, seasonal predictors
#    winter_start = ee.Date(startDate)
#    winter_end = ee.Date(startDate).advance(3,'month')
#    spring_start = ee.Date(startDate).advance(3,'month')
#    spring_end = ee.Date(startDate).advance(6,'month')
#    summer_start = ee.Date(startDate).advance(6,'month')
#    summer_end = ee.Date(startDate).advance(9,'month')
#    fall_start = ee.Date(startDate).advance(9,'month')
#    fall_end = ee.Date(endDate)
#
#  # Aggregate seasonal info for each variable of interest (potEvap neglected purposefully)
#    seasonal_precip = add_seasonal_info(NLDAS_precip,"Precip","total_precipitation")
#    seasonal_temp = add_seasonal_info(NLDAS_temp,"Temp","temperature")
#    seasonal_humid = add_seasonal_info(NLDAS_humid,"Humidity","specific_humidity")
#
#    waterYear_start = ee.Date(startDate).advance(10,'month')
#    waterYear_end = waterYear_start.advance(1,'year')
#
#  #========================================================
#  # Aggregate Other Covariates
#  #========================================================
#
#  # Vegetative Continuous Fields
#    meanVCF = VCF.filterDate(startDate, endDate)\
#                 .mean()
#    
##     VCF_qc.filterDate(startDate, endDate) \
##                       .mean()
#
#  # Filter Precip by water year to get total precip annually
#
#    waterYearTot = NLDAS_precip.filterDate(waterYear_start,waterYear_end) \
#                                 .sum()
#
#  # Find mean EVI per year:
#    maxEVI = EVI.filterDate(startDate,endDate) \
#                  .mean() \
#                  .rename(['Mean_EVI'])
#
#  #Find mean NDVI per year:
#    maxNDVI = NDVI.filterDate(startDate,endDate) \
#                    .mean() \
#                    .rename(["Mean_NDVI"])
#
#  # Find flashiness per year by taking a Per-pixel Standard Deviation:
#    flashiness_yearly = ee.Image(pekel_monthly_water.filterDate(startDate,endDate) \
#                                                      .reduce(ee.Reducer.sampleStdDev()) \
#                                                      .select(["water_stdDev"])) \
#                                                      .rename("Flashiness")
#
#  # Find max LST per year:
#    maxLST = LST.max().rename(["Max_LST_Annual"])
#
#  # Find mean GPP per year:
#    maxGPP = GPP_QC.filterDate(startDate,endDate) \
#                      .mean() \
#                      .rename(['Mean_GPP','QC'])
#
#  # All banded images that don't change over time
#    static_input_bands = sw_occurrence.addBands(DEM.select("elevation")) \
#                                          .addBands(srtmChili) \
#                                          .addBands(topoDiv) \
#                                          .addBands(footprint)
#
#  # Construct huge banded image
#    banded_image = static_input_bands \
#                          .addBands(srcImg = maxLST, names = ["Max_LST_Annual"]) \
#                          .addBands(srcImg = maxGPP, names = ["Mean_GPP"]) \
#                          .addBands(srcImg =  maxNDVI, names = ["Mean_NDVI"]) \
#                          .addBands(srcImg = maxEVI, names = ["Mean_EVI"]) \
#                          .addBands(meanVCF.select("Percent_Tree_Cover")) \
#                          .addBands(seasonal_precip) \
#                          .addBands(flashiness_yearly) \
#                          .set("system:time_start",startDate)
#
#    return banded_image.unmask()
#
##//========================================================




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









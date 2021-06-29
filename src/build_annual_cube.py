import ee

#Build Big Raster Image
## Import assets
# MODIS Mission
modusGlobal = ee.ImageCollection("MODIS/006/MYD11A2")

# Primary Productivity
GPP = ee.ImageCollection("UMT/NTSG/v2/LANDSAT/GPP")

# Surface water
pikelSurfaceWater = ee.Image("JRC/GSW1_1/GlobalSurfaceWater")

# Elevation
DEM = ee.Image("USGS/NED")

# Enhanced Vegetation Index and NDVI
modusVeg = ee.ImageCollection("MODIS/006/MYD13A2")

# Heat Isolation Load
CHILI = ee.Image("CSP/ERGo/1_0/Global/SRTM_CHILI")

# Topographic Diversity
topoDiversity = ee.Image("CSP/ERGo/1_0/Global/ALOS_topoDiversity")

# Vegetation Continuous Field product - percent tree cover, etc
VCF = ee.ImageCollection("MODIS/006/MOD44B")

# Human Modification index
gHM = ee.ImageCollection("CSP/HM/GlobalHumanModification")

# Climate information
NLDAS = ee.ImageCollection("NASA/NLDAS/FORA0125_H002")

# Shape file containing Country Boundaries
countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")

# Shape file containing HUC polygons
HUC = ee.FeatureCollection("USGS/WBD/2017/HUC12")

# Dynamic Surface Water metric
pekel_monthly_water = ee.ImageCollection("JRC/GSW1_2/MonthlyHistory")

# Static surface water metric
pekel_static_water = ee.ImageCollection('JRC/GSW1_2/MonthlyRecurrence')

## Select features, etc
#========================================================
#Rename Bands and select bands, etc
#========================================================


NLDAS_precip = NLDAS.select("total_precipitation");
NLDAS_temp = NLDAS.select("temperature");
NLDAS_humid = NLDAS.select("specific_humidity");
NLDAS_potEvap = NLDAS.select("potential_evaporation");


CHILI = CHILI.rename(['Heat_Insolation_Load'])
srtmChili = CHILI.select('Heat_Insolation_Load');
topoDiversity = topoDiversity.rename(["Topographic_Diversity"])
topoDiv = topoDiversity.select("Topographic_Diversity")
footprint = ee.Image(gHM.first().select("gHM"));

# Surface water occurrence
sw_occurrence = pekel_static_water\
                      .select('monthly_recurrence')\
                      .mean()\
                      .rename(['SurfaceWaterOccurrence'])\
                      .unmask()

## Mask features by quality control bands
GPP_QC = GPP.map(gpp_qc);


LST = modusGlobal.map(lst_qc) \
                 .select("LST_Day_1km");

modusVeg_QC = modusVeg.map(modusQC)
EVI = modusVeg_QC.select("EVI")
NDVI = modusVeg_QC.select("NDVI")

VCF_qc = VCF.map(VCFqc)


#========================================================
# Define Point Joins such that each HUC contains a list of observational data:
#========================================================
distFilter = ee.Filter.intersects(**{
  'leftField': '.geo', 
  'rightField': '.geo', 
  'maxError': 100
});

pointJoin = ee.Join.saveAll(**{
  'matchesKey': 'Points',
});


## Annual Cube function
#========================================================
# "Builder Function" -- processes each annual variable into a list of images
#========================================================

def build_annual_cube(d):
    # Set start and end dates for filtering time dependent predictors (SR, NDVI, Phenology)
      # Advance startDate by 1 to begin with to account for water year (below)
    startDate = (ee.Date(d).advance(1.0,'year').millis()) ## FIXME: Why do we advance a year? this give 2003-2019 instead of 2002-2018
    endDate = ee.Date(d).advance(2.0,'year').millis()

  #========================================================
  #Define function to compute seasonal information for a given variable
  #========================================================
    def add_seasonal_info(imgCol,name,bandName):
        winter = imgCol.filterDate(winter_start,winter_end)
        spring = imgCol.filterDate(spring_start,spring_end)
        summer = imgCol.filterDate(summer_start,summer_end)
        fall = imgCol.filterDate(fall_start,fall_end)

        winter_tot = winter.sum()
        spring_tot = spring.sum()
        summer_tot = summer.sum()
        fall_tot = fall.sum()

        winter_max = winter.max()
        winter_min = winter.min()
        spring_max = spring.max()
        spring_min = spring.min()
        summer_max = summer.max()
        summer_min = summer.min()
        fall_max = fall.max()
        fall_min = fall.min()

        winter_diff = winter_max.subtract(winter_min)
        spring_diff = spring_max.subtract(spring_min)
        summer_diff = summer_max.subtract(summer_min)
        fall_diff = fall_max.subtract(fall_min)

        names = ['winter_total'+name,'spring_total'+name,'summer_total'+name,
                      'fall_total'+name]

        return winter_tot.addBands([spring_tot,summer_tot,fall_tot]) \
                         .rename(names)

  # Set up Seasonal dates for precip, seasonal predictors
    winter_start = ee.Date(startDate)
    winter_end = ee.Date(startDate).advance(3,'month')
    spring_start = ee.Date(startDate).advance(3,'month')
    spring_end = ee.Date(startDate).advance(6,'month')
    summer_start = ee.Date(startDate).advance(6,'month')
    summer_end = ee.Date(startDate).advance(9,'month')
    fall_start = ee.Date(startDate).advance(9,'month')
    fall_end = ee.Date(endDate)

  # Aggregate seasonal info for each variable of interest (potEvap neglected purposefully)
    seasonal_precip = add_seasonal_info(NLDAS_precip,"Precip","total_precipitation")
    seasonal_temp = add_seasonal_info(NLDAS_temp,"Temp","temperature")
    seasonal_humid = add_seasonal_info(NLDAS_humid,"Humidity","specific_humidity")

    waterYear_start = ee.Date(startDate).advance(10,'month')
    waterYear_end = waterYear_start.advance(1,'year')

  #========================================================
  # Aggregate Other Covariates
  #========================================================

  # Vegetative Continuous Fields
    meanVCF = VCF.filterDate(startDate, endDate)\
                 .mean()
    
#     VCF_qc.filterDate(startDate, endDate) \
#                       .mean()

  # Filter Precip by water year to get total precip annually

    waterYearTot = NLDAS_precip.filterDate(waterYear_start,waterYear_end) \
                                 .sum()

  # Find mean EVI per year:
    maxEVI = EVI.filterDate(startDate,endDate) \
                  .mean() \
                  .rename(['Mean_EVI'])

  #Find mean NDVI per year:
    maxNDVI = NDVI.filterDate(startDate,endDate) \
                    .mean() \
                    .rename(["Mean_NDVI"])

  # Find flashiness per year by taking a Per-pixel Standard Deviation:
    flashiness_yearly = ee.Image(pekel_monthly_water.filterDate(startDate,endDate) \
                                                      .reduce(ee.Reducer.sampleStdDev()) \
                                                      .select(["water_stdDev"])) \
                                                      .rename("Flashiness")

  # Find max LST per year:
    maxLST = LST.max().rename(["Max_LST_Annual"])

  # Find mean GPP per year:
    maxGPP = GPP_QC.filterDate(startDate,endDate) \
                      .mean() \
                      .rename(['Mean_GPP','QC'])

  # All banded images that don't change over time
    static_input_bands = sw_occurrence.addBands(DEM.select("elevation")) \
                                          .addBands(srtmChili) \
                                          .addBands(topoDiv) \
                                          .addBands(footprint)

  # Construct huge banded image
    banded_image = static_input_bands \
                          .addBands(srcImg = maxLST, names = ["Max_LST_Annual"]) \
                          .addBands(srcImg = maxGPP, names = ["Mean_GPP"]) \
                          .addBands(srcImg =  maxNDVI, names = ["Mean_NDVI"]) \
                          .addBands(srcImg = maxEVI, names = ["Mean_EVI"]) \
                          .addBands(meanVCF.select("Percent_Tree_Cover")) \
                          .addBands(seasonal_precip) \
                          .addBands(flashiness_yearly) \
                          .set("system:time_start",startDate)

    return banded_image.unmask()

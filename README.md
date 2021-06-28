# AIS
Aquatic Invasive Species (AIS) establishment prediction toolset


## Pathname convention:
<pre>
AIS/
  |- notebook.ipynb
  |- datasets/
    |- points/
      |- # Presence and absence data points
    |- hucs/
      |- # .geojson files, by state
    |- training/
      |- # Location for script output files, used fo traiing ML alg
    |- decade/
      |- # Data grouped by decade. We're still figuring out what this is 
</pre>






### Note on pathnames:
  The script expects to find relevant files in the locations as specified above.
  You can find the paths in the "Define Modular Variables" block.

  We also currently store GEE hosted files at 'users/kjchristensen93/'


# plot study regions close up

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # basemap tools
from matplotlib.patches import Polygon
from netCDF4 import Dataset # reads netCDF file
import numpy as np
import cmocean # oceanogrpahy colorscales - https://matplotlib.org/cmocean/
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import math

def harversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # harversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2.)**2. + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2.)**2.
    c = 2. * math.asin(math.sqrt(a))
    km = 6371. * c # radius of earth
    return km

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

# Setup maps
##############################################################################
m = Basemap(projection='merc', llcrnrlat=-37.15, urcrnrlat=-35.35,\
        llcrnrlon=149.11, urcrnrlon=151.34, lat_ts=20, resolution='h')
##############################################################################

# Study region plot
##############################################################################
fig = plt.figure(figsize=(3,3))
plt.tight_layout()
# draw stuff
m.drawcoastlines(color='black', linewidth=0.7)
m.fillcontinents(color='#A0A0A0')
# # add grid
# parallels = np.arange(-81.,0,1.)
# m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
# meridians = np.arange(10.,351.,1.)
# m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')

# add bathymetry
nc_file = '/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset/naroom_avg_01461.nc'
fh = Dataset(nc_file, mode='r')
lats = fh.variables['lat_rho'][:] 
lons = fh.variables['lon_rho'][:]
bath = fh.variables['h'][:]
bath = bath/1000
#cs = m.pcolor(lons,lats,np.squeeze(bath), latlon = True ,vmin=0, vmax=5, cmap=cmocean.cm.deep)
cs_coast = m.contourf(lons, lats, np.squeeze(bath), list(frange(0, 0.501, 0.5)), cmap='Blues_r', latlon=True)
cs_offsh = m.contourf(lons, lats, np.squeeze(bath), list(frange(0.5, 4.001, 3.5)), cmap='Reds_r', latlon=True)
CS1 = m.contour(lons,lats,np.squeeze(bath),[0.5], colors='k', latlon=True, linestyles='dashed', alpha=0.9)
CS2 = m.contour(lons,lats,np.squeeze(bath),[4], colors='k', latlon=True, linestyles='dashed', alpha=0.9)
plt.clabel(CS1, inline=1, fmt='500m')
plt.clabel(CS2, inline=1, fmt='4000m')

# add montague Island
lon = 150.2269
lat = -36.25201
x,y = m(lon, lat)
m.plot(x, y, '+', markersize=14, mew=2, color='#ff0000')

##############################################################################
# fig2 = plt.figure()
# # map2 inset
# m2.drawcoastlines(color='black', linewidth=0.7)
# m2.fillcontinents(color='#A0A0A0')
# # add grid
# parallels = np.arange(-81.,0,.5)
# m2.drawparallels(parallels,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
# meridians = np.arange(10.,351.,.5)
# m2.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
# mark_inset(fig, fig2, loc1=2, loc2=4, fc="none", ec="0.5")

plt.show()

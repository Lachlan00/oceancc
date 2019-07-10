# plot study regions

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

# Setup maps
##############################################################################
m = Basemap(projection='merc', llcrnrlat=-44.050653, urcrnrlat=-32.453367,\
        llcrnrlon=145.996456, urcrnrlon=154.457344, lat_ts=20, resolution='h')
# m2 = Basemap(projection='merc', llcrnrlat=-37.15, urcrnrlat=-35.35,\
#         llcrnrlon=149.11, urcrnrlon=151.34, lat_ts=20, resolution='h')
##############################################################################

# Study region plot
##############################################################################
fig = plt.figure(figsize=(8,8))
plt.tight_layout()
# draw stuff
m.drawcoastlines(color='black', linewidth=0.7)
m.fillcontinents(color='#A0A0A0')
# add grid
parallels = np.arange(-81.,0,2.)
m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
meridians = np.arange(10.,351.,2.)
m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')

# add bathymetry
nc_file = '/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset/naroom_avg_01461.nc'
fh = Dataset(nc_file, mode='r')
lats = fh.variables['lat_rho'][:] 
lons = fh.variables['lon_rho'][:]
bath = fh.variables['h'][:]
bath = bath/1000
cs = m.pcolor(lons,lats,np.squeeze(bath), latlon = True ,vmin=0, vmax=5, cmap=cmocean.cm.deep)
cbar = plt.colorbar()
CS1 = m.contour(lons,lats,np.squeeze(bath),[0.5], colors='k', latlon=True, linestyles='dashed', alpha=0.9)
CS2 = m.contour(lons,lats,np.squeeze(bath),[4], colors='k', latlon=True, linestyles='dashed', alpha=0.9)
cbar.ax.invert_yaxis()
cbar.ax.set_ylabel('Depth (km)', rotation=270, labelpad=16, size=12)



# add region zone
# setup montague polygon
xmin1, xmax1, ymin1, ymax1 = 149.11, 151.34, -37.15, -35.35
x1,y1 = m(xmin1,ymin1) 
x2,y2 = m(xmin1,ymax1) 
x3,y3 = m(xmax1,ymax1) 
x4,y4 = m(xmax1,ymin1)
# mont region polygon
p = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)], facecolor='none',
	edgecolor='#ff0000',linewidth=2,ls='dashed', zorder=10)
plt.gca().add_patch(p)
# full region
xmin1, xmax1, ymin1, ymax1 = 147.996456, 152.457344, -38.050653, -34.453367
x1,y1 = m(xmin1,ymin1) 
x2,y2 = m(xmin1,ymax1) 
x3,y3 = m(xmax1,ymax1) 
x4,y4 = m(xmax1,ymin1)
p = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)], facecolor='none',
	edgecolor='#ff0000',linewidth=2, zorder=10)
#plt.gca().add_patch(p)

# add montague Island
lon = 150.2269
lat = -36.25201
x,y = m(lon, lat)
m.plot(x, y, '+', markersize=14, mew=2, color='#ff0000')

# add distance lines
ymax_lat = lats[0, -1]
ymin_lat = lats[-1, -1]
ymax_lon = lons[0, -1]
ymin_lon = lons[-1, -1]
xmax_lat = lats[0, -1]
xmin_lat = lats[0, 0]
xmax_lon = lons[0, -1]
xmin_lon = lons[0, 0]
ydistance = str(int(round(harversine(ymin_lon, ymin_lat, ymax_lon, ymax_lat))))
xdistance = str(int(round(harversine(xmin_lon, xmin_lat, xmax_lon, xmax_lat))))
ymax_lon, ymax_lat = m(ymax_lon, ymax_lat)
ymin_lon, ymin_lat = m(ymin_lon, ymin_lat)
xmax_lon, xmax_lat = m(xmax_lon, xmax_lat)
xmin_lon, xmin_lat = m(xmin_lon, xmin_lat)    
# define linear equations to offset lines

# draw arrows
# y
xoff = 20000
yoff = -10000
plt.annotate(s='', xy=(ymax_lon+xoff,ymax_lat+yoff), xytext=(ymin_lon+xoff,ymin_lat+yoff), 
	arrowprops=dict(arrowstyle='<->', color='#505050', linewidth=1))
plt.text(ymax_lon+110000, ymax_lat+310000, ydistance + ' km', rotation=73)
# x
xoff = 0
yoff = -30000
plt.annotate(s='', xy=(xmax_lon+xoff,xmax_lat+yoff), xytext=(xmin_lon+xoff,xmin_lat+yoff), 
	arrowprops=dict(arrowstyle='<->', color='#505050', linewidth=1))
plt.text(xmin_lon+200000, xmin_lat-150000, xdistance + ' km', rotation=-20)



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

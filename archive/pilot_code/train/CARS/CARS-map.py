# map coral sea using CARS data

import numpy as np
import pandas as pd
from netCDF4 import Dataset # reads netCDF file
import math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # basemap tools
from matplotlib.patches import Polygon
from datetime import datetime
import cmocean # oceanogrpahy colorscales - https://matplotlib.org/cmocean/

# __SETUP__
xmin1, xmax1, ymin1, ymax1 = 155, 160, -28.5, -22.5 # north box
xmin2, xmax2, ymin2, ymax2 = 155, 160, -46, -41 # south box

def add(lst, obj, index): return lst[:index] + [obj] + lst[index:]

def plot_day(data, vmin, vmax, style):
    # map setup
    fig = plt.figure()
    fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # draw stuff
    m.drawcoastlines(color='black', linewidth=0.7)
    m.fillcontinents(color='#A0A0A0')
    # plot salt
    cs = m.pcolor(lons, lats, np.squeeze(data), latlon=True, vmin=vmin, vmax=vmax, cmap=style)
    # make rectangle
    xmin1, xmax1, ymin1, ymax1 = 155, 160, -28, -22.5
    xmin2, xmax2, ymin2, ymax2 = 155, 160, -46, -41.5
    x1,y1 = m(xmin1,ymin1) 
    x2,y2 = m(xmin1,ymax1) 
    x3,y3 = m(xmax1,ymax1) 
    x4,y4 = m(xmax1,ymin1)
    p = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)], facecolor='none',edgecolor='red',linewidth=2.5,zorder=10,ls='dashed')
    plt.gca().add_patch(p)
    plt.text(x1+(x3-x1)/2, (y1+(y3-y1)/2)-30000, 'A', color='red', size=20, ha='center', va='center', weight='bold') 
    x1,y1 = m(xmin2,ymin2) 
    x2,y2 = m(xmin2,ymax2) 
    x3,y3 = m(xmax2,ymax2) 
    x4,y4 = m(xmax2,ymin2)
    p = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)], facecolor='none',edgecolor='yellow',linewidth=2.5,zorder=10,ls='dashed') 
    plt.gca().add_patch(p)
    plt.text(x1+(x3-x1)/2, (y1+(y3-y1)/2)-30000, 'B', color='yellow', size=20, ha='center', va='center', weight='bold')   
    # stop axis from being cropped
    plt.tight_layout()
    return fig

# Temp
file_path = '/Users/lachlanphillips/PhD_Large_Data/CARS/temperature_cars2009a.nc'
fh = Dataset(file_path, mode='r')
T_mean = fh.variables['mean'][0,:,:]
T_an_cos = fh.variables['an_cos'][0,:,:]
T_an_sin = fh.variables['an_sin'][0,:,:]
T_sa_cos = fh.variables['sa_cos'][0,:,:]
T_sa_sin = fh.variables['sa_sin'][0,:,:]
# Salt
file_path = '/Users/lachlanphillips/PhD_Large_Data/CARS/salinity_cars2009a.nc'
fh = Dataset(file_path, mode='r')
S_mean = fh.variables['mean'][0,:,:]
S_an_cos = fh.variables['an_cos'][0,:,:]
S_an_sin = fh.variables['an_sin'][0,:,:]
S_sa_cos = fh.variables['sa_cos'][0,:,:]
S_sa_sin = fh.variables['sa_sin'][0,:,:]

# list of days to extract
months = [1,2,3,4,5,6,7,8,9,10,11,12]
month_days = [15,45,74,105,135,166,196,227,258,288,319,349]

# get lons and lats
lons = fh.variables['lon'][:]
lats = fh.variables['lat'][:]
# make into 2d arrays
lons = np.tile(lons, 331)
lons = np.reshape(lons,(331,721))
lats = np.repeat(lats,721)
lats = np.reshape(lats,(331,721))

# Setup map
##############################################################################
m = Basemap(projection='merc', llcrnrlat=-47.050653, urcrnrlat=-13.453367,\
        llcrnrlon=144.296456, urcrnrlon=168.457344, lat_ts=20, resolution='h')
##############################################################################

day = int(month_days[1])
dt = datetime(2001, int(months[1]), 15, 0, 0, 0)
t = 2*math.pi*(day/366)
T = T_mean + T_an_cos*math.cos(t) + T_an_sin*math.sin(t) + T_sa_cos*math.cos(2*t) + T_sa_sin*math.sin(2*t)
S = S_mean + S_an_cos*math.cos(t) + S_an_sin*math.sin(t) + S_sa_cos*math.cos(2*t) + S_sa_sin*math.sin(2*t)

# make plots

fig_temp = plot_day(T, 13, 32, cmocean.cm.thermal)
# add grid
parallels = np.arange(-81.,0,5.)
m.drawparallels(parallels,labels=[False,True,False,True], linewidth=1, dashes=[3,3], color='#707070')
meridians = np.arange(10.,351.,5.)
m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
# colorbar
cbaxes = fig_temp.add_axes([0.21, 0.039, 0.035, .923])  # This is the position for the colorbar
cbar = plt.colorbar(cax=cbaxes)
cbar.ax.set_ylabel('Mean Temperature (℃)', labelpad=16, size=10)
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_label_position('left')

fig_salt = plot_day(S, 34.4, 35.9, cmocean.cm.haline)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Mean Salinity (PSU)', rotation=270, labelpad=19, size=10)
# add grid
parallels = np.arange(-81.,0,5.)
m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
meridians = np.arange(10.,351.,5.)
m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
plt.show()































# check day differences

import numpy as np
import pandas as pd
from netCDF4 import Dataset # reads netCDF file
import math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # basemap tools
from matplotlib.patches import Polygon
from datetime import datetime
from datetime import timedelta
# __SETUP__
xmin1, xmax1, ymin1, ymax1 = 155, 160, -28.5, -22.5 # north box
xmin2, xmax2, ymin2, ymax2 = 155, 160, -46, -41 # south box

def add(lst, obj, index): return lst[:index] + [obj] + lst[index:]

def plot_day(data, vmin, vmax, style, title):
    # map setup
    fig = plt.figure()
    fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # draw stuff
    m.drawcoastlines() # comment out when using shapefile
    m.fillcontinents(color='black')
    # plot salt
    cs = m.pcolor(lons, lats, np.squeeze(data), latlon=True, vmin=vmin, vmax=vmax, cmap=style)
    # make rectangle
    xmin1, xmax1, ymin1, ymax1 = 155, 160, -28, -22.5
    xmin2, xmax2, ymin2, ymax2 = 155, 160, -46, -41.5
    x1,y1 = m(xmin1,ymin1) 
    x2,y2 = m(xmin1,ymax1) 
    x3,y3 = m(xmax1,ymax1) 
    x4,y4 = m(xmax1,ymin1)
    p = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)], facecolor='none',edgecolor='red',linewidth=2)
    plt.gca().add_patch(p) 
    x1,y1 = m(xmin2,ymin2) 
    x2,y2 = m(xmin2,ymax2) 
    x3,y3 = m(xmax2,ymax2) 
    x4,y4 = m(xmax2,ymin2)
    p = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)], facecolor='none',edgecolor='yellow',linewidth=2) 
    plt.gca().add_patch(p)  
    # plot colourbar
    plt.colorbar()
    # datetime title
    plt.title(title)
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

day = 5
t = 2*math.pi*(day/366)
T1 = T_mean + T_an_cos*math.cos(t) + T_an_sin*math.sin(t) + T_sa_cos*math.cos(2*t) + T_sa_sin*math.sin(2*t)
S1 = S_mean + S_an_cos*math.cos(t) + S_an_sin*math.sin(t) + S_sa_cos*math.cos(2*t) + S_sa_sin*math.sin(2*t)

day = 360
t = 2*math.pi*(day/366)
T2 = T_mean + T_an_cos*math.cos(t) + T_an_sin*math.sin(t) + T_sa_cos*math.cos(2*t) + T_sa_sin*math.sin(2*t)
S2 = S_mean + S_an_cos*math.cos(t) + S_an_sin*math.sin(t) + S_sa_cos*math.cos(2*t) + S_sa_sin*math.sin(2*t)

plt.close("all")
title = 'Regional - Temperature (Celcius) | Day: 5'
fig1 = plot_day(T1, 14, 32, 'plasma', title)
fig2 = plot_day(T2, 14, 32, 'plasma', title)
title = 'Regional - Salinity (PSU) | Day: 360'
fig3 = plot_day(S1, 35.2, 35.9, 'viridis', title)
fig3 = plot_day(S2, 35.2, 35.9, 'viridis', title)

plt.show()











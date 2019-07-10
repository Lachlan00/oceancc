# map coral sea using CARS data

import numpy as np
import pandas as pd
from netCDF4 import Dataset # reads netCDF file
import math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # basemap tools
from matplotlib.patches import Polygon
from datetime import datetime
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

df_list = []

plot_bool = str(input('Show plots? (y/n): '))

for k in range (0, len(months)):
    print('Extractign data from month '+str(months[k]).zfill(2)+' of 12...')
    # create map for desired time (e.g. mid Feb [day 45])
    day = int(month_days[k])
    dt = datetime(2001, int(months[k]), 15, 0, 0, 0)
    t = 2*math.pi*(day/366)
    T = T_mean + T_an_cos*math.cos(t) + T_an_sin*math.sin(t) + T_sa_cos*math.cos(2*t) + T_sa_sin*math.sin(2*t)
    S = S_mean + S_an_cos*math.cos(t) + S_an_sin*math.sin(t) + S_sa_cos*math.cos(2*t) + S_sa_sin*math.sin(2*t)

    # make plot
    if plot_bool == 'y':
        plt.close("all")
        title = 'Regional - Temperature (Celcius) | Day: ' + str(day)
        fig1 = plot_day(T, 14, 32, 'plasma', title)
        title = 'Regional - Salinity (PSU) | Day: ' + str(day)
        fig2 = plot_day(S, 35.2, 35.9, 'viridis', title)
        print('Extracting data for month '+str(months[k]))
        input("Press Enter to continue...")
        plt.show()

    #############
    # NORTH BOX #
    #############

    # get values inside north box
    point_zip = zip(lats.ravel(), lons.ravel())
    point_list = []
    j = 0
    for i in point_zip:
        if ymin1 <= i[0] <= ymax1 and xmin1 <= i[1] <=xmax1:
            point_list.append(j)
        j += 1
    # convert to grid coordinates
    lon_grid = []
    lat_grid = []
    j = 0
    for i in point_list:
        lon_grid = add(lon_grid, int(i%721), j)
        lat_grid = add(lat_grid, int(i/721), j)
    # get salinity and temperature data
    n_temp = list(T[lat_grid,lon_grid].data)
    n_salt = list(S[lat_grid,lon_grid].data)
    dfn = {'temp':n_temp, 'salt':n_salt}
    dfn = pd.DataFrame(data=dfn)
    dfn['datetime'] = dt
    dfn['class'] = 'EAC'

    #############
    # SOUTH BOX #
    #############

    # get values inside north box
    point_zip = zip(lats.ravel(), lons.ravel())
    point_list = []
    j = 0
    for i in point_zip:
        if ymin2 <= i[0] <= ymax2 and xmin2 <= i[1] <=xmax2:
            point_list.append(j)
        j += 1
    # convert to grid coordinates
    lon_grid = []
    lat_grid = []
    j = 0
    for i in point_list:
        lon_grid = add(lon_grid, int(i%721), j)
        lat_grid = add(lat_grid, int(i/721), j)
    # get salinity and temperature data
    s_temp = list(T[lat_grid,lon_grid].data)
    s_salt = list(S[lat_grid,lon_grid].data)
    dfs = {'temp':s_temp, 'salt':s_salt}
    dfs = pd.DataFrame(data=dfs)
    dfs['datetime'] = dt
    dfs['class'] = 'BS'

    # add to dataframe list
    df_list.append(dfn)
    df_list.append(dfs)

# concat all collected data
data_out = pd.concat(df_list).reset_index(drop=True)
# output
output_fn = '/Users/lachlanphillips/Dropbox/PhD/Analysis/ocean_classification/train/CARS/CARS_output/CARS_data.csv'
data_out.to_csv(output_fn, index=False)






























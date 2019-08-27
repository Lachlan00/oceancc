# Train current tracking model using data from the CSIRO Atlas of Regional Seas (CARS) 

import numpy as np
import pandas as pd
from netCDF4 import Dataset # reads netCDF file
import math
from datetime import datetime
from datetime import timedelta
import cmocean
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from progressbar import ProgressBar

# Hack to fix missing PROJ4 env var for basemap
import os
import conda
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap

# local modules
from data_processes import *

#####################
# Plot source boxes #
#####################
def plot_box_map(temp, lons, lats, sourceboxA, sourceboxB):
    """
    Plot source boxes for a visual check
    """
    print('Plotting boxes...')
    all_lats = sourceboxA[2:4] + sourceboxB[2:4]
    all_lons = sourceboxA[0:2] + sourceboxB[0:2]
    map_region = [min(all_lons)-5, max(all_lons)+5, min(all_lats)-5, max(all_lats)+5]
    m = Basemap(projection='merc', llcrnrlat=min(all_lats)-5, urcrnrlat=max(all_lats)+5,\
        llcrnrlon=min(all_lons)-5, urcrnrlon=max(all_lons)+5, lat_ts=20, resolution='h')
    fig = plt.figure()
    fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # draw stuff
    m.drawcoastlines(color='black', linewidth=0.7)
    m.fillcontinents(color='#A0A0A0')
    # colour map of tmeperature
    # get max temps
    vmin, vmax = minmax_in_region(temp, lons, lats, map_region)
    cs = m.pcolor(lons, lats, np.squeeze(temp), latlon=True, vmin=vmin, 
        vmax=vmax, cmap=cmocean.cm.thermal)
    # make boxes
    # A
    x1,y1 = m(sourceboxA[0],sourceboxA[2]) 
    x2,y2 = m(sourceboxA[0],sourceboxA[3]) 
    x3,y3 = m(sourceboxA[1],sourceboxA[3]) 
    x4,y4 = m(sourceboxA[1],sourceboxA[2])
    p = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)], facecolor='none',edgecolor='black',linewidth=2.5,zorder=10,ls='dashed')
    plt.gca().add_patch(p)
    plt.text(x1+(x3-x1)/2, (y1+(y3-y1)/2)-30000, 'A', color='black', size=20, ha='center', va='center', weight='bold') 
    # B
    x1,y1 = m(sourceboxB[0],sourceboxB[2]) 
    x2,y2 = m(sourceboxB[0],sourceboxB[3]) 
    x3,y3 = m(sourceboxB[1],sourceboxB[3]) 
    x4,y4 = m(sourceboxB[1],sourceboxB[2])
    p = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)], facecolor='none',edgecolor='black',linewidth=2.5,zorder=10,ls='dashed') 
    plt.gca().add_patch(p)
    plt.text(x1+(x3-x1)/2, (y1+(y3-y1)/2)-30000, 'B', color='black', size=20, ha='center', va='center', weight='bold')   
    # add grid
    parallels = np.arange(-81.,0,5.)
    m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    meridians = np.arange(10.,351.,5.)
    m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    # plot colourbar
    plt.colorbar()
    # title
    plt.title('Current Source Boxes')
    # stop axis from being cropped
    plt.tight_layout()
    return fig


#################################
# Train model from CARS dataset #
#################################
def train_CARS(input_directory, output_directory, sourceboxA, sourceboxB, plot_boxes=False):
    """
    Sourcebox format = [xmin, xmax, ymin, ymax]
    """
    # check if training data already exists
    if os.path.exists(output_directory+'training_data.csv'):
        if yes_or_no('"'+output_directory+'training_data.csv" already exists. Would you like to produce a new training dataset?'):
            print('"'+output_directory+'training_data.csv" will be overwirtten.')
        else:
            print('Using previously extracted dataset.')
            return

    # start program
    print('\nExtracting training data from CARS model...')
    print('Loading data and constructing model...')
    # Read in data
    # Temperature
    file_path = input_directory + 'temperature_cars2009a.nc'
    fh = Dataset(file_path, mode='r')
    T_mean = fh.variables['mean'][0,:,:]
    T_an_cos = fh.variables['an_cos'][0,:,:]
    T_an_sin = fh.variables['an_sin'][0,:,:]
    T_sa_cos = fh.variables['sa_cos'][0,:,:]
    T_sa_sin = fh.variables['sa_sin'][0,:,:]
    # Salinity
    file_path = input_directory + 'salinity_cars2009a.nc'
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

    # check boxes
    if plot_boxes:
        plot_box_map(T_mean, lons, lats, sourceboxA, sourceboxB)
        plt.show()

    # make list to hold df
    df_list = []
    # set a random year to isolate days
    base_dt = datetime(2001, 1, 1, 0, 0, 0)
    # set variable names for reasier code
    xmin1, xmax1, ymin1, ymax1 = sourceboxA
    xmin2, xmax2, ymin2, ymax2 = sourceboxB

    # setup progress bar
    print('Extracting data from CARS model')
    pbar = ProgressBar(max_value=365)
    pbar.update(0)

    # BOX A
    # get values inside box A
    point_zip = zip(lats.ravel(), lons.ravel())
    point_listA = []
    j = 0
    for i in point_zip:
        if ymin1 <= i[0] <= ymax1 and xmin1 <= i[1] <= xmax1:
            point_listA.append(j)
        j += 1

     # BOX B
    # get values inside box B
    point_zip = zip(lats.ravel(), lons.ravel())
    point_listB = []
    j = 0
    for i in point_zip:
        if ymin2 <= i[0] <= ymax2 and xmin2 <= i[1] <= xmax2:
            point_listB.append(j)
        j += 1
    
    # iterate through each modelled day in CARS and get data
    for k in range (1, 366):
        # get data for desired time (e.g. mid Feb [day 45])
        day = int(k)
        td = timedelta(k-1)
        dt = base_dt + td
        t = 2*math.pi*(day/366)
        T = T_mean + T_an_cos*math.cos(t) + T_an_sin*math.sin(t) + T_sa_cos*math.cos(2*t) + T_sa_sin*math.sin(2*t)
        S = S_mean + S_an_cos*math.cos(t) + S_an_sin*math.sin(t) + S_sa_cos*math.cos(2*t) + S_sa_sin*math.sin(2*t)

        # A - convert to grid coordinates
        lon_grid = []
        lat_grid = []
        j = 0
        for i in point_listA:
            lon_grid = add(lon_grid, int(i%721), j)
            lat_grid = add(lat_grid, int(i/721), j)
        # get salinity and temperature data
        n_temp = list(T[lat_grid,lon_grid].data)
        n_salt = list(S[lat_grid,lon_grid].data)
        dfA = {'temp':n_temp, 'salt':n_salt}
        dfA = pd.DataFrame(data=dfA)
        dfA['datetime'] = dt
        dfA['class'] = 'A'

        # B - convert to grid coordinates
        lon_grid = []
        lat_grid = []
        j = 0
        for i in point_listB:
            lon_grid = add(lon_grid, int(i%721), j)
            lat_grid = add(lat_grid, int(i/721), j)
        # get salinity and temperature data
        s_temp = list(T[lat_grid,lon_grid].data)
        s_salt = list(S[lat_grid,lon_grid].data)
        dfB = {'temp':s_temp, 'salt':s_salt}
        dfB = pd.DataFrame(data=dfB)
        dfB['datetime'] = dt
        dfB['class'] = 'B'

        # add to dataframe list
        df_list.append(dfA)
        df_list.append(dfB)

        # update progress bar
        pbar.update(k)

    # extract and save
    data_out = pd.concat(df_list).reset_index(drop=True)
    output_fn = output_directory + 'training_data.csv'
    data_out.to_csv(output_fn, index=False)
    print('\ndata saved to: '+output_fn)














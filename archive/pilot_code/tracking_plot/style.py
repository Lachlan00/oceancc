# example plot of tracking

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # basemap tools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta # for working with datetimes
from netCDF4 import Dataset # reads netCDF file
from os import listdir
from os.path import isfile, join
import argparse
from matplotlib.patches import Polygon
import cmocean # oceanogrpahy colorscales - https://matplotlib.org/cmocean/

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

# __SETUP__

time_step = 27
file_idx = 29

def grab_sst_time(time_idx):
    """
    gets datetime object for sst map projection
    """
    dtcon_days = time[time_idx]
    dtcon_start = datetime(1990,1,1) # This is the "days since" part
    dtcon_delta = timedelta(dtcon_days/24/60/60) # Create a time delta object from the number of days
    dtcon_offset = dtcon_start + dtcon_delta # Add the specified number of days to 1990
    frame_time = dtcon_offset
    return frame_time

# __Get_Data__

# get list of files in data directory
in_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset"
file_ls = [f for f in listdir(in_directory) if isfile(join(in_directory, f))]
file_ls = list(filter(lambda x:'naroom_avg' in x, file_ls))
file_ls = sorted(file_ls)

# load in file to get lats and lons and data for plottin
frame_idx = time_step
nc_file = in_directory + '/' + file_ls[file_idx]
fh = Dataset(nc_file, mode='r')
lats = fh.variables['lat_rho'][:] 
lons = fh.variables['lon_rho'][:]
temp = fh.variables['temp'][frame_idx,29,:,:]
salt = fh.variables['salt'][frame_idx,29,:,:]
time = fh.variables['ocean_time'][:]

# __Make_Plots__

# plot function
def fun_plot(data, vmin, vmax, cmap, title):
    fig = plt.figure()
    plt.tight_layout()
    # draw stuff
    m.drawcoastlines(color='black', linewidth=0.7)
    m.fillcontinents(color='#A0A0A0')
    cs = m.contourf(lons, lats, np.squeeze(data), list(frange(vmin, vmax, 0.4)), cmap=cmap, latlon=True, vmin=vmin, vmax=vmax, extend='both')
    #cs = m.pcolor(lons,lats,np.squeeze(data), latlon = True ,vmin=vmin, vmax=vmax, cmap=cmap)

    return fig

# Setup map
##############################################################################
m = Basemap(projection='merc', llcrnrlat=-38.050653, urcrnrlat=-35.653367,\
        llcrnrlon=147.996456, urcrnrlon=152.457344, lat_ts=20, resolution='h')
##############################################################################

fig_temp = fun_plot(temp, 14, 23, cmocean.cm.thermal, 'Regional - Temperature (℃)\n')


plt.show()























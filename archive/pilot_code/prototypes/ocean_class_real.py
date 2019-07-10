"""
Ocean classification algorithm
------------------------------
Experimental program to classify two different water masses 
based on temperature and salinity profiles. 
TEST: use real data
"""

# import modules
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import math
import matplotlib.pyplot as plt
import pylab
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap # basemap tools
import argparse
import datetime
from datetime import datetime, timedelta # for working with datetimes
from netCDF4 import Dataset # reads netCDF file
from os import listdir
from os.path import isfile, join

# __Setup__

parser = argparse.ArgumentParser(description=__doc__)

# add a positional writable file argument
parser.add_argument('input_csv_file', type=argparse.FileType('r'))
# parser.add_argument('output_csv_file', type=argparse.FileType('w')) 

args = parser.parse_args()

#Read GPS tracks csv files
csv_data = pd.read_csv(args.input_csv_file, parse_dates = ['datetime'], 
						infer_datetime_format = True) #Read as DateTime obsject

# set colour scale variables
temp_min = 14
temp_max = 24
salt_min = 35.3
salt_max = 35.7
prob_max = 1.
prob_min = 0.


# for getting time data
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

def plot_temp(temp, time_idx):
    """
    Make maps of temperature and salinity
    """

    # make frame__idx an integer to avoid slicing errors
    frame_idx = int(time_idx)

    # get 'frame_time'
    frame_time = grab_sst_time(frame_idx)

    # map setup
    fig = plt.figure()
    fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # Setup the map
    m = Basemap(projection='merc', llcrnrlat=-38.050653, urcrnrlat=-34.453367,\
            llcrnrlon=147.996456, urcrnrlon=152.457344, lat_ts=20, resolution='h') # full range
    # draw stuff
    m.drawcoastlines() # comment out when using shapefile
    m.fillcontinents(color='black')
    # plot salt
    cs = m.pcolor(lons,lats,np.squeeze(temp), latlon = True ,vmin=temp_min, vmax=temp_max, cmap='plasma')
    # plot colourbar
    plt.colorbar()
    # datetime title
    plt.title('Regional - Temperature (Celcius)\n' + frame_time.strftime("%Y-%m-%d %H:%M:%S") + ' | ' + str(fname) + '_idx: ' + str(frame_idx))
    # stop axis from being cropped
    plt.tight_layout()

def plot_salt(salt, time_idx):
    """
    Make maps of temperature and salinity
    """

    # make frame__idx an integer to avoid slicing errors
    frame_idx = int(time_idx)

    # get 'frame_time'
    frame_time = grab_sst_time(frame_idx)

    # map setup
    fig = plt.figure()
    fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # Setup the map
    m = Basemap(projection='merc', llcrnrlat=-38.050653, urcrnrlat=-34.453367,\
            llcrnrlon=147.996456, urcrnrlon=152.457344, lat_ts=20, resolution='h') # full range
    # draw stuff
    m.drawcoastlines() # comment out when using shapefile
    m.fillcontinents(color='black')
    # plot salt
    cs = m.pcolor(lons,lats,np.squeeze(salt), latlon = True ,vmin=salt_min, vmax=salt_max, cmap='viridis')
    # plot colourbar
    plt.colorbar()
    # datetime title
    plt.title('Regional - Salinity (PSU)\n' + frame_time.strftime("%Y-%m-%d %H:%M:%S") + ' | ' + str(fname) + '_idx: ' + str(frame_idx))
    # stop axis from being cropped
    plt.tight_layout()

def plot_prob(probs, time_idx):
    """
    Make maps of temperature and salinity
    """

    # make frame__idx an integer to avoid slicing errors
    frame_idx = int(time_idx)

    # get 'frame_time'
    frame_time = grab_sst_time(frame_idx)

    # map setup
    fig = plt.figure()
    fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # Setup the map
    m = Basemap(projection='merc', llcrnrlat=-38.050653, urcrnrlat=-34.453367,\
            llcrnrlon=147.996456, urcrnrlon=152.457344, lat_ts=20, resolution='h') # full range
    # draw stuff
    m.drawcoastlines() # comment out when using shapefile
    m.fillcontinents(color='black')
    # plot salt
    cs = m.pcolor(lons,lats,np.squeeze(probs), latlon = True ,vmin=prob_min, vmax=prob_max, cmap='bwr')
    # plot colourbar
    plt.colorbar()
    # datetime title
    plt.title('Regional - EAC Probability\n' + frame_time.strftime("%Y-%m-%d %H:%M:%S") + ' | ' + str(fname) + '_idx: ' + str(frame_idx))
    # stop axis from being cropped
    plt.tight_layout()


# create data points for training algorithm
# 1 = classA (EAC), 0 = classB (TS)
var1 = list(csv_data['temp'])
var2 = list(csv_data['salt'])
water_class = list(csv_data['class'])

# make data frame
train_data = {'var1': var1, 'var2': var2, 'class': water_class}
train_data = pd.DataFrame(data=train_data)

# replace current data strings with binary integers
train_data['class'] = train_data['class'].replace(to_replace='EAC', value=1)
train_data['class'] = train_data['class'].replace(to_replace='BS', value=0)

# fit logistic regression to the training data
lr_model = LogisticRegression()
lr_model = lr_model.fit(train_data[['var1','var2']], np.ravel(train_data[['class']]))

# load in array for testing
# get list of files in data directory
directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset"
file_ls = [f for f in listdir(directory) if isfile(join(directory, f))]
file_ls = list(filter(lambda x:'naroom_avg' in x, file_ls))

# get file for testing **
fn = file_ls[212]
file_path = directory + "/" + fn
fname = str(fn)[11:16]

# load data
fh = Dataset(file_path, mode='r')

# random time for testing **
time_idx = 25

# extract data
lats = fh.variables['lat_rho'][:]
lons = fh.variables['lon_rho'][:]
time = fh.variables['ocean_time'][:]
temp = fh.variables['temp'][time_idx,29,:,:] 
salt = fh.variables['salt'][time_idx,29,:,:]

# ravel to 1D array
temp1d = temp.ravel()
salt1d = salt.ravel()

data = {'var1': temp1d, 'var2': salt1d}
data = pd.DataFrame(data=data)
data = data.fillna(9999)

# calculate probabilities
probs = lr_model.predict_proba(data[['var1','var2']])
prob_TSW, prob_EAC = zip(*probs)

# convert tuples to list
prob_EAC = list(prob_EAC)
# sub back in nans
prob_EAC = [x if x != 1. else np.nan for x in prob_EAC]
# make 1D array
prob_EAC = np.asarray(prob_EAC)
# make 2D array
prob_EAC = np.reshape(prob_EAC, (-1, 165))

# make plots
fig = plot_temp(temp, time_idx)
# fig = plot_salt(salt, time_idx)
fig = plot_prob(prob_EAC, time_idx)

plt.show()

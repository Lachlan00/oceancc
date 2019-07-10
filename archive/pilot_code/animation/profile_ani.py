# Animate T-S profiles

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
from scipy import stats
import cmocean
from random import randint

# animation modules
import moviepy.editor as mpy # creates animation
from moviepy.video.io.bindings import mplfig_to_npimage # converts map to numpy array
from matplotlib.backends.backend_agg import FigureCanvasAgg # draws canvas so that map can be converted

# how many files
start = 0
end = 2

# for 
prob_ls = np.empty(0)
time_ls = np.empty(0)
temp_ls = np.empty(0)
salt_ls = np.empty(0)

# range for floats
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

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

def make_frame(j):
    plt.close("all")
    frame_idx = int(j)
    frame_time = grab_sst_time(int(j))
    # get data
    temp = fh.variables['temp'][j,29,:,:] 
    salt = fh.variables['salt'][j,29,:,:]
    # remove masks
    temp = np.ma.filled(temp.astype(float), np.nan)
    salt = np.ma.filled(salt.astype(float), np.nan)
    # ravel
    temp = temp.ravel()
    salt = salt.ravel()
    # remove NaNs
    temp = temp[~np.isnan(temp)]
    salt = salt[~np.isnan(salt)]
    # add data to lists
    fig = plt.figure()
    plt.scatter(salt, temp, marker='.', s=2)
    plt.xlabel('Salinity (PSU)', fontsize=14)
    plt.ylabel('Temperature ($^\circ$C)', fontsize=14)
    plt.title('Frame Temperature-Salinty Profile\n' + frame_time.strftime("%Y-%m-%d %H:%M:%S") + ' | ' + 
        str(fname) + '_idx: ' + str(frame_idx).zfill(2))
    axes = plt.gca()
    axes.set_xlim([35.1,35.8])
    axes.set_ylim([11,28])

    #convert to array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    frame = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return frame

# __Grab_Data__

# get list of files in data directory
in_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset"
file_ls = [f for f in listdir(in_directory) if isfile(join(in_directory, f))]
file_ls = list(filter(lambda x:'naroom_avg' in x, file_ls))
file_ls = sorted(file_ls)

# set output directory
out_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/gifs"

# load one file to get lats and lons and data for initial plot
nc_file = in_directory + '/' + file_ls[1]
fh = Dataset(nc_file, mode='r')

# get all the probability arrays
# itterate through all files
idx = 0
for i in range(start, end):
    # import file
    nc_file = in_directory + '/' + file_ls[i]
    fh = Dataset(nc_file, mode='r')
    print('Grabbing data from: '+file_ls[i]+' | '+str(i+1).zfill(3)+' of '+ str(len(file_ls)))
    fname = str(file_ls[i])[11:16]
    # extract time
    time = fh.variables['ocean_time'][:]

    # set gif names
    out_fn = out_directory+'/full_run/profile/'+'TS_'+str(i+1).zfill(3)+'.gif'

    # build animation
    frame_count = len(time)
    animation = mpy.VideoClip(make_frame, duration=frame_count)
    animation.write_gif(out_fn, fps=1)

    # close file
    fh.close()

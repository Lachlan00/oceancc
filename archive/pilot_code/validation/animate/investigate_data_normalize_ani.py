# Investigative plots

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

# normalisation modules
from sklearn import preprocessing

# animation modules
import moviepy.editor as mpy # creates animation
from moviepy.video.io.bindings import mplfig_to_npimage # converts map to numpy array
from matplotlib.backends.backend_agg import FigureCanvasAgg # draws canvas so that map can be converted

# __SETUP__

parser = argparse.ArgumentParser(description=__doc__)
# add a positional writable file argument
parser.add_argument('input_csv_file', type=argparse.FileType('r'))
# parser.add_argument('output_csv_file', type=argparse.FileType('w')) 
args = parser.parse_args()

# how many files
start = 0
end = 277 

def make_frame(j):
    plt.close("all")
    frame_idx = int(j)
    frame_time = grab_sst_time(int(j))
    DoY_val = int(frame_time.day)
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
    # make dataframe
    test_data = {'temp': temp, 'salt': salt}
    test_data = pd.DataFrame(data=test_data)
    # scale data
    test_data['temp'] = scaler_temp.transform(test_data[['temp']])
    test_data['salt'] = scaler_salt.transform(test_data[['salt']])
    # get prob
    prob = get_prob(test_data['temp'], test_data['salt'], DoY_val, lr_model)
    # add data to lists
    fig = plt.figure()
    plt.scatter(salt, temp, marker='.', s=2, c=prob, cmap='bwr')
    plt.xlabel('Salinity (PSU)', fontsize=14)
    plt.ylabel('Temperature ($^\circ$C)', fontsize=14)
    cbar = plt.colorbar()
    cbar.ax.set_yticks([0.0,0.5,1.0])
    # cbar.ax.set_yticklabels(['0', '0.5', '1'])
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

# get probs
def get_prob(temp1d, salt1d, DoY_val, lr_model):
    # make data frame and replace NaNs
    data = {'var1': temp1d, 'var2': salt1d}
    data = pd.DataFrame(data=data)
    data['DoY'] = DoY_val
    # calculate probabilities
    probs = lr_model.predict_proba(data[['var1','var2','DoY']])
    prob_TSW, prob_EAC = zip(*probs)
    # convert tuples to list
    prob_EAC = list(prob_EAC)

    return prob_EAC

# __Make_Model__

# Read in classification training data
csv_data = pd.read_csv(args.input_csv_file, parse_dates = ['datetime'], 
                        infer_datetime_format = True) #Read as DateTime obsject

# add "Day of year" (DoY) to dataset 
csv_data['DoY'] = [int(x.day) for x in csv_data['datetime']]

# make training dataset model
# create data points for training algorithm
# 1 = classA (EAC), 0 = classB (TS)
var1 = list(csv_data['temp'])
var2 = list(csv_data['salt'])
DoY = list(csv_data['DoY'])
water_class = list(csv_data['class'])
# make data frame
train_data = {'var1': var1, 'var2': var2, 'DoY': DoY, 'class': water_class}
train_data = pd.DataFrame(data=train_data)
# replace current data strings with binary integers
train_data['class'] = train_data['class'].replace(to_replace='EAC', value=1)
train_data['class'] = train_data['class'].replace(to_replace='BS', value=0)

###################
# Standarise Data #
###################

# get fits to be used for later scaling
scaler_temp = preprocessing.StandardScaler().fit(train_data[['var1']])
scaler_salt = preprocessing.StandardScaler().fit(train_data[['var2']])
# scale training dataset
train_data['var1'] = scaler_temp.transform(train_data[['var1']])
train_data['var2'] = scaler_salt.transform(train_data[['var2']])

###################

# fit logistic regression to the training data
lr_model = LogisticRegression()
lr_model = lr_model.fit(train_data[['var1','var2','DoY']], np.ravel(train_data[['class']]))

# __Grab_Data__

# get list of files in data directory
in_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset"
file_ls = [f for f in listdir(in_directory) if isfile(join(in_directory, f))]
file_ls = list(filter(lambda x:'naroom_avg' in x, file_ls))
file_ls = sorted(file_ls)

# set output directory
out_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/gifs"

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








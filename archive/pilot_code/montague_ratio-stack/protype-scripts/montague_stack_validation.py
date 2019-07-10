"""
Ocean classification algorithm
------------------------------
Experimental program to classify two different water masses 
based on temperature and salinity profiles. 
"""

print('_____Extracting_EAC_Ratio_Stack_____')

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
from itertools import compress # for filteriung using boolean masks

# animation modules
import moviepy.editor as mpy # creates animation
from moviepy.video.io.bindings import mplfig_to_npimage # converts map to numpy array
from matplotlib.backends.backend_agg import FigureCanvasAgg # draws canvas so that map can be converted

# __Setup__

parser = argparse.ArgumentParser(description=__doc__)
# add a positional writable file argument
parser.add_argument('input_csv_file', type=argparse.FileType('r'))
# parser.add_argument('output_csv_file', type=argparse.FileType('w')) 
args = parser.parse_args()

# set range for data to collect
xmin = 149.669301
xmax = 150.784499
ymin = -36.701671
ymax = -35.802349

# set colour scale variables
prob_max = 1.
prob_min = 0.

# lists to collect ratio data
yEAC = []
yTSW = []
yEACr = []
yTSWr = []
xtime = []
xmonth = []

# How many NetCDF files to use (set the range)
start = 0
end = 277 # for 277 files (zero index but range() does not include last number)

# randmoness seed (for shuffling)
np.random.seed(420)

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

def grab_prob(time_idx):
    # make frame__idx an integer to avoid slicing errors
    frame_idx = int(time_idx)
    # get 'frame_time'
    frame_time = grab_sst_time(frame_idx)

    # get list of temperature and salinity values from subset
    temp = []
    salt = []

    # append values
    for i, j in point_list:
    	temp.append(fh.variables['temp'][frame_idx,29,i,j])
    	salt.append(fh.variables['salt'][frame_idx,29,i,j])

    data = {'var1': temp, 'var2': salt}
    data = pd.DataFrame(data=data)
    # remove masked floats
    data = data[data.var1 >= 1]

    # calculate probabilities
    probs = lr_model.predict_proba(data[['var1','var2']])
    prob_TSW, prob_EAC = zip(*probs)
    # convert tuples to list
    prob_EAC = list(prob_EAC)
    # sub back in nans
    prob_EAC = [x if x != 0.0 else np.nan for x in prob_EAC]
    # make 1D array
    prob_EAC = np.asarray(prob_EAC)
    # make 2D array (not required in subset as not being plotted)
    # prob_EAC = np.reshape(prob_EAC, (-1, 165))

    # calulcate ratio metric
    count_EAC = np.count_nonzero(prob_EAC > 0.5)
    count_TSW = np.count_nonzero(prob_EAC < 0.5)
    # add to lists
    yEAC.append(count_EAC)
    yTSW.append(count_TSW)
    yEACr.append(count_EAC/473)
    yTSWr.append(count_TSW/473)
    xtime.append(frame_time)
    xmonth.append(frame_time.month)

# __Make_Model__

print('\nBuilding predictive model from training data...')

# Read in classification training data
csv_data = pd.read_csv(args.input_csv_file, parse_dates = ['datetime'], 
                        infer_datetime_format = True) #Read as DateTime obsject

# make training dataset model
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
# remove 25% of data for modle validation 
train_data = train_data.iloc[np.random.permutation(np.arange(len(train_data)))].reset_index(drop=True) # shuffle dataset
split_idx = int(len(train_data) - int(len(train_data))*0.7) # point to split df (70/30%)
valid_data = train_data.iloc[split_idx:, :].reset_index(drop=True)
train_data = train_data.iloc[:split_idx, :].reset_index(drop=True)

# fit logistic regression to the training data
lr_model = LogisticRegression()
lr_model = lr_model.fit(train_data[['var1','var2']], np.ravel(train_data[['class']]))

# __Validate_Model__ 
# NOTE: Temp method, later use cross validation

print('\nValidating model...')

valid_probs = lr_model.predict_proba(valid_data[['var1','var2']])
valid_TSW, valid_EAC = zip(*valid_probs)
valid_EAC = list(valid_EAC)
valid_df = {'prob': valid_EAC, 'class': valid_data['class']}
valid_df = pd.DataFrame(data=valid_df)
valid_df['result'] = [1 if x >= 0.5 else 0 for x in valid_df['prob']]

# calculate accuracy
valid_result = 0
for idx, row in valid_df.iterrows():
    if int(row['class']) == int(row['result']):
        valid_result = valid_result + 1
valid_result = valid_result/len(valid_df['class'])
print('\nAccuracy of model is calculated to be at ' + str("%.2f" % (valid_result*100)) + ' %')
print('Later version of program will use \"Cross Validation\"\n')
# "%.2f" % value - rounds printing to 2 decimal places
# Will later add cross validation to the model

# __Main_Program__

# get list of files in data directory
in_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset"
file_ls = [f for f in listdir(in_directory) if isfile(join(in_directory, f))]
file_ls = list(filter(lambda x:'naroom_avg' in x, file_ls))
file_ls = sorted(file_ls)

# get lats and lons
nc_file = in_directory + '/' + file_ls[0]
fh = Dataset(nc_file, mode='r')
lats = fh.variables['lat_rho'][:] 
lons = fh.variables['lon_rho'][:]
# combine to list of tuples
point_tuple = zip(lats.ravel(), lons.ravel())
point_list = []

j = 0
# iterate over tuple points and keep every point that is in box
for i in point_tuple:
	if ymin <= i[0] <= ymax and xmin <= i[1] <=xmax:
		point_list.append(j)
	j = j + 1

# make point list into tuple list of array coordinates
eta_rho = []
xi_rho = []
for i in point_list:
	eta_rho.append(int(i/165))
	xi_rho.append(int(i%165))

point_list = zip(eta_rho,xi_rho)
point_list = set(point_list) # due to zip behaviour in Python3

# get data for each file (may take a while)
for i in range(start, end):
    # import file
    nc_file = in_directory + '/' + file_ls[i]
    fh = Dataset(nc_file, mode='r')
    print('getting data from file: '+file_ls[i]+' | '+str(i+1)+' of '+ str(len(file_ls)))
    # fname = str(file_ls[i])[11:16]

    # extract time
    time = fh.variables['ocean_time'][:]

    # get data
    for i in range(0, len(time)):
        grab_prob(i)
    # close file
    fh.close()

# make dataframe from ratio metric lists
df_ratio  = {'yEAC': yEAC, 'yTSW': yTSW, 'yEACr': yEACr, 'yTSWr': yTSWr, 'xtime': xtime, 'xmonth': xmonth}
df_ratio = pd.DataFrame(data=df_ratio)
# remove duplicates
df_ratio = df_ratio.drop_duplicates('xtime').reset_index()
print(df_ratio)

# save to csv
print('_____Please_Wait_____')
print('Writing ocean data to file...')
output_fn = '../montague_ratio-stack/output/' + 'mont-stack2_' + str(start).zfill(3) + '-' + str(end).zfill(3) + '.csv' 
df_ratio.to_csv(output_fn, index=False)

# make stack plot of results
print('Displaying plot - close plot to continue...')
plt.close('all')
fig, ax = plt.subplots()
ax.stackplot(xtime, yTSWr, yEACr)
plt.show()

print('_____Program_End_____')


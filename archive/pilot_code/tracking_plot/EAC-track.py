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

# __SETUP__

time_step = 29
file_idx = 101

parser = argparse.ArgumentParser(description=__doc__)
# add a positional writable file argument
parser.add_argument('input_csv_file', type=argparse.FileType('r'))
# parser.add_argument('output_csv_file', type=argparse.FileType('w')) 
args = parser.parse_args()

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
# fit logistic regression to the training data
lr_model = LogisticRegression()
lr_model = lr_model.fit(train_data[['var1','var2','DoY']], np.ravel(train_data[['class']]))

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

# get probability array
frame_idx = int(frame_idx)
frame_time = grab_sst_time(frame_idx)
DoY_val = int(frame_time.day)
# ravel to 1D arrays
temp1d = temp.ravel()
salt1d = salt.ravel()
# make data frame and replace NaNs
data = {'var1': temp1d, 'var2': salt1d}
data = pd.DataFrame(data=data)
data['DoY'] = DoY_val
data = data.fillna(-9999)
# calculate probabilities
probs = lr_model.predict_proba(data[['var1','var2','DoY']])
prob_TSW, prob_EAC = zip(*probs)
# convert tuples to list
prob_EAC = list(prob_EAC)
# sub back in nans
prob_EAC = [x if x != 0.0 else np.nan for x in prob_EAC]
# make 1D array
prob_EAC = np.asarray(prob_EAC)
# make 2D array
prob = np.reshape(prob_EAC, (-1, 165))

# __Make_Plots__
out_dir = '/Users/lachlanphillips/Dropbox/PhD/THESIS/figures/results/'

# plot function
def fun_plot(data, vmin, vmax, cmap, title):
    fig = plt.figure()
    plt.tight_layout()
    # draw stuff
    m.drawcoastlines(color='black', linewidth=0.7)
    m.fillcontinents(color='#A0A0A0')
    # add grid
    parallels = np.arange(-81.,0,1.)
    m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    meridians = np.arange(10.,351.,2.)
    m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    cs = m.pcolor(lons,lats,np.squeeze(data), latlon = True ,vmin=vmin, vmax=vmax, cmap=cmap)
    # plot colourbar
    plt.colorbar()
    plt.title(title + frame_time.strftime("%Y-%m-%d %H:%M:%S"))

    return fig

# Setup map
##############################################################################
m = Basemap(projection='merc', llcrnrlat=-38.050653, urcrnrlat=-34.453367,\
        llcrnrlon=147.996456, urcrnrlon=152.457344, lat_ts=20, resolution='h')
##############################################################################

fig_temp = fun_plot(temp, 15, 23, cmocean.cm.thermal, 'Regional - Temperature (℃)\n')
fig_salt = fun_plot(salt, 35.45, 35.8, cmocean.cm.haline, 'Regional - Salinity (PSU)\n')
fig_prob = fun_plot(prob, 0, 1, cmocean.cm.balance, 'Regional - EAC Probability\n')

fig_temp.savefig(out_dir + 'out-salt.png')
fig_salt.savefig(out_dir + 'out-temp.png')
fig_prob.savefig(out_dir + 'out-prob.png')

plt.show()























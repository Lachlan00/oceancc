"""
Ocean classification algorithm
------------------------------
Experimental program to classify two different water masses 
based on temperature and salinity profiles. 
TEST: use real data
"""

print('_____Building_Animation_____')

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

# set colour scale variables
prob_max = 1.
prob_min = 0.

# lists to collect ratio data
yEAC = []
xtime = []

# add to list witout append
def add(lst, obj, index): return lst[:index] + [obj] + lst[index:]

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

def plot_prob(time_idx):
    """
    Make maps of temperature and salinity
    """
    # close previous figures to free up memory
    plt.close('all')

    # make frame__idx an integer to avoid slicing errors
    frame_idx = int(time_idx)

    # get 'frame_time'
    frame_time = grab_sst_time(frame_idx)

    # Get probs
    temp = fh.variables['temp'][frame_idx,29,:,:] 
    salt = fh.variables['salt'][frame_idx,29,:,:]
    # ravel to 1D array
    temp1d = temp.ravel()
    salt1d = salt.ravel()
    # make data frame and replace NaNs
    data = {'var1': temp1d, 'var2': salt1d}
    data = pd.DataFrame(data=data)
    data = data.fillna(-9999)
    # calculate probabilities
    probs = lr_model.predict_proba(data[['var1','var2']])
    prob_TSW, prob_EAC = zip(*probs)
    # convert tuples to list
    prob_EAC = list(prob_EAC)
    # sub back in nans
    prob_EAC = [x if x != 0.0 else np.nan for x in prob_EAC]
    # make 1D array
    prob_EAC = np.asarray(prob_EAC)
    # make 2D array
    prob_EAC = np.reshape(prob_EAC, (-1, 165))

    # calulcate ratio metric
    count_EAC = np.count_nonzero(prob_EAC > 0.5)
    count_TSW = np.count_nonzero(prob_EAC < 0.5)
    
    # add to lists # GLOBAL
    global yEAC, xtime, list_count
    yEAC, xtime = add(yEAC, count_EAC, list_count), add(xtime, frame_time, list_count)
    list_count = list_count + 1

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
    cs = m.pcolor(lons,lats,np.squeeze(prob_EAC), latlon = True ,vmin=prob_min, vmax=prob_max, cmap='bwr')
    # plot colourbar
    plt.colorbar()
    # datetime title
    plt.title('Regional - EAC Probability\n' + frame_time.strftime("%Y-%m-%d %H:%M:%S") + ' | ' + 
        str(fname) + '_idx: ' + str(frame_idx).zfill(2))
    # stop axis from being cropped
    plt.tight_layout()

    #convert to array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    frame = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return frame

# __Make_Model__

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
# fit logistic regression to the training data
lr_model = LogisticRegression()
lr_model = lr_model.fit(train_data[['var1','var2']], np.ravel(train_data[['class']]))

# __Main_Program__

# get list of files in data directory
in_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset"
file_ls = [f for f in listdir(in_directory) if isfile(join(in_directory, f))]
file_ls = list(filter(lambda x:'naroom_avg' in x, file_ls))
file_ls = sorted(file_ls)

# set output directory
out_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/gifs"

# set range
start = 0
end = 2 # for 277 files (zero index)
# make animation for each file (will take over 10 hours)
list_count = 0
for i in range(start, end):
    # import file
    nc_file = in_directory + '/' + file_ls[i]
    fh = Dataset(nc_file, mode='r')
    print('Building animations with file: '+file_ls[i]+' | '+str(i+1)+' of '+ str(len(file_ls)))
    fname = str(file_ls[i])[11:16]

    # set gif names
    out_prob = out_directory+'/full_run/class_index/'+'prob'+str(i+1).zfill(3)+'.gif'

    # extract data
    lats = fh.variables['lat_rho'][:] 
    lons = fh.variables['lon_rho'][:]
    time = fh.variables['ocean_time'][:]

    # make animations
    frame_count = len(time)
    animation = mpy.VideoClip(plot_prob, duration=frame_count)
    animation.write_gif(out_prob, fps=1)
    # close file
    fh.close()

# make dataframe from ratio metric lists
df_ratio  = {'yEAC': yEAC, 'xtime': xtime}
df_ratio = pd.DataFrame(data=df_ratio)
df_ratio['yTSW'] = [12121 - x for x in df_ratio['yEAC']]
df_ratio['yEACr'] = [x/12121 for x in df_ratio['yEAC']]
df_ratio['yTSWr'] = [x/12121 for x in df_ratio['yTSW']]
df_ratio['xmonth'] = [x.month for x in df_ratio['xtime']]
# remove duplicates
df_ratio = df_ratio.drop_duplicates('xtime').reset_index()
print(df_ratio)
print(df_ratio)

# save to csv
print('_____Please_Wait_____')
print('Writing ocean data to file...')
dt = datetime.now()
dt = dt.strftime('%Y%m%d-%H%M')
output_fn = '../outputs/classification_stack/' + 'index-stack_' + str(start).zfill(3) + '-' + str(end).zfill(3) + '_' + dt + '.csv' 
df_ratio.to_csv(output_fn, index=False)

# make stack plot of results
print('Displaying plot - close plot to continue...')
plt.close('all')
fig, ax = plt.subplots()
ax.stackplot(df_ratio['xtime'], df_ratio['yTSWr'], df_ratio['yEACr'])
plt.show()

print('_____Program_End_____')
print('Tip: To increase fame speed, use the following commands to use imagemagick from terminal:')
print('convert -delay 25x100 gif_name.gif gif_name.gif')
print('NetCDF animation builder v-2.0 does not currently support subplots.')
print('For now use "https://ezgif.com/combine" to join the gifs.')


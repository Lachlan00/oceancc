# min max finder

import argparse
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from netCDF4 import Dataset # reads netCDF file
from datetime import datetime, timedelta # for working with datetimes

# __SETUP__

parser = argparse.ArgumentParser(description=__doc__)
# add a positional writable file argument
parser.add_argument('input_csv_file', type=argparse.FileType('r'))
# parser.add_argument('output_csv_file', type=argparse.FileType('w')) 
args = parser.parse_args()

# Read in classification training data
data = pd.read_csv(args.input_csv_file, parse_dates = ['datetime'], 
                        infer_datetime_format = True) #Read as DateTime obsject

start = 0
end = 277

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

# find max min
CARS_temp_max = data.temp.max()
CARS_temp_min = data.temp.min()
CARS_salt_max = data.salt.max()
CARS_salt_min = data.salt.min()

print('######################')
print('# CARS min-max data: #')
print('######################\n')
print('temp-max: ' + str(CARS_temp_max))
print('temp-min: ' + str(CARS_temp_min))
print('salt-max: ' + str(CARS_salt_max))
print('salt-min: ' + str(CARS_salt_min))
print('\n######################\n')

# ROMS data
max_temp = []
min_temp = []
max_salt = []
min_salt = []

# get list of files in data directory
in_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset"
file_ls = [f for f in listdir(in_directory) if isfile(join(in_directory, f))]
file_ls = list(filter(lambda x:'naroom_avg' in x, file_ls))
file_ls = sorted(file_ls)

for i in range(start, end):
    # import file
    nc_file = in_directory + '/' + file_ls[i]
    fh = Dataset(nc_file, mode='r')
    print('Grabbing data from: '+file_ls[i]+' | '+str(i+1).zfill(3)+' of '+ str(len(file_ls)), end='\r')

    # extract time
    time = fh.variables['ocean_time'][:]

    # iterate through all time steps
    for j in range(0, len(time)):
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
        max_temp.append(temp.max())
        min_temp.append(temp.min())
        max_salt.append(salt.max())
        min_salt.append(salt.min())

    # close file
    fh.close()

# find max min
ROMS_temp_max = max(max_temp)
ROMS_temp_min = min(min_temp)
ROMS_salt_max = max(max_salt)
ROMS_salt_min = min(min_salt)

print('')
print('\n######################')
print('# ROMS min-max data: #')
print('######################\n')
print('temp-max: ' + str(ROMS_temp_max))
print('temp-min: ' + str(ROMS_temp_min))
print('salt-max: ' + str(ROMS_salt_max))
print('salt-min: ' + str(ROMS_salt_min))
print('\n######################\n')

if CARS_temp_max >= ROMS_temp_max:
	temp_max = CARS_temp_max
else:
	temp_max = ROMS_temp_max
if CARS_temp_min <= ROMS_temp_min:
	temp_min = CARS_temp_min
else:
	temp_min = ROMS_temp_min
if CARS_salt_max >= ROMS_salt_max:
	salt_max = CARS_salt_max
else:
	salt_max = ROMS_salt_max
if CARS_salt_min <= ROMS_salt_min:
	salt_min = CARS_salt_min
else:
	salt_min = ROMS_salt_min

print('\n######################')
print('# Tota min-max data: #')
print('######################\n')
print('temp-max: ' + str(temp_max))
print('temp-min: ' + str(temp_min))
print('salt-max: ' + str(salt_max))
print('salt-min: ' + str(salt_min))
print('\n######################\n')

"""
######################
# Tota min-max data: #
######################

temp-max: 27.3994477596
temp-min: 9.53612435175
salt-max: 35.8450737
salt-min: 33.8999481201

######################
"""



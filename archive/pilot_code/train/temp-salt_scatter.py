# create random scatter plots of temperature and salinity a bunch of random time slices. 
 
from netCDF4 import Dataset # reads netCDF file
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pylab
from datetime import datetime, timedelta #for working with datetimes
from random import randint
import numpy as np

# Define functions
def scatter_plot(temp, salt, time_value, fig_no):
	"""
	Plots temperature and salinity on a scatter plot
	"""
	fig = plt.figure(fig_no)
	plt.scatter(salt, temp, s=5, alpha=0.3)
	plt.ylabel('Temperature', fontsize=14)
	plt.xlabel('Salinity', fontsize=14)
	plt.title('Montague Region - Temperature/Salinity\n' + time_value.strftime("%Y-%m-%d %H:%M:%S"))
	return fig

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

# get list of files in data directory
directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset"
file_ls = [f for f in listdir(directory) if isfile(join(directory, f))]
file_ls = list(filter(lambda x:'naroom_avg' in x, file_ls))

# make random plots
# set randomness seed
plot_num = 20
np.random.seed(1010)
rnd_file = np.random.randint(len(file_ls), size=plot_num)
rnd_times = np.random.randint(29, size=plot_num)

# save plot to list
plot_ls = list(range(0, plot_num))
# make the plots
for i in range(0, plot_num):
	# grab file
	file_no = rnd_file[i]
	file_path = directory + "/" + file_ls[file_no]
	# grab time
	time_idx = rnd_times[i]
	fh = Dataset(file_path, mode='r')
	# extract time
	time = fh.variables['ocean_time'][:]
	time_value = grab_sst_time(time_idx)
	# extract temperature and salinity
	temp = fh.variables['temp'][time_idx,29,:,:] 
	salt = fh.variables['salt'][time_idx,29,:,:] 
	plot_ls[i] = scatter_plot(temp, salt, time_value, i)

plt.show()






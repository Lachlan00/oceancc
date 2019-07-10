"""
Script to calculate mean anomalies for winter/summer for each year

STEPS:
1 - Extract all data into 1 dataframe
2 - Calculate means for each season/year pairs
3 - Calulate deviation from mean
4 - Make plots
"""

# animate_class_fast_new.py

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # basemap tools
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import argparse
from scipy import stats
import cmocean
from netCDF4 import Dataset # reads netCDF file

# __SETUP__

parser = argparse.ArgumentParser(description=__doc__)
# add a positional writable file argument
parser.add_argument('input_csv_file', type=argparse.FileType('r'))
# parser.add_argument('output_csv_file', type=argparse.FileType('w')) 
args = parser.parse_args()

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
bath = fh.variables['h'][:]

df=pd.read_csv(args.input_csv_file, sep=',',header=None)
total_std = df.values
# range for floats
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def make_plot(data,lons,lats,vmin,vmax,cmap):
    fig = plt.figure()
    # draw stuff
    m.drawcoastlines(color='black', linewidth=0.7)
    m.fillcontinents(color='#A0A0A0')
    # plot color
    # cs = m.pcolor(lons,lats,np.squeeze(data), latlon = True ,vmin=vmin, vmax=vmax, cmap=cmap)
    cs = m.contourf(lons, lats, np.squeeze(data), list(frange(vmin, vmax, 0.2)), cmap=cmap, latlon=True, vmin=vmin, vmax=vmax, extend='both')
    cbar = plt.colorbar(ticks=[20,21,22,23,24,25])
    cbar.set_label('Variance (σ)', rotation=270, labelpad=20, size=13)
    # datetime title
    plt.tight_layout()
    parallels = np.arange(-81.,0,.5)
    m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    meridians = np.arange(10.,351.,.5)
    m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    CS = m.contour(lons,lats,np.squeeze(bath),[500], colors='k', latlon=True, linestyles='dashed', alpha=0.6)
    plt.clabel(CS, inline=True, fmt='500 m', fontsize=11)


    return fig

##############################################################################
m = Basemap(projection='merc', llcrnrlat=-37.15, urcrnrlat=-35.35,\
        llcrnrlon=149.11, urcrnrlon=151.34, lat_ts=20, resolution='h')
##############################################################################

fig3 = make_plot(total_std, lons, lats, 20, 25, cmocean.cm.thermal)

plt.show()


"""
The plots don't give any useful information... need to rethink this... 
"""

















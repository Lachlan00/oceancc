# vector field test

from netCDF4 import Dataset # reads netCDF file
from mpl_toolkits.basemap import Basemap # basemap tools
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np

# for testing
i = 42
t = 10

# get list of files in data directory
in_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset"
file_ls = [f for f in listdir(in_directory) if isfile(join(in_directory, f))]
file_ls = list(filter(lambda x:'naroom_avg' in x, file_ls))
file_ls = sorted(file_ls)

nc_file = in_directory + '/' + file_ls[i]
fh = Dataset(nc_file, mode='r')
fname = str(file_ls[i])[11:16]

# extract data
lats = fh.variables['lat_rho'][:] 
lons = fh.variables['lon_rho'][:]
u = fh.variables['u'][t,29,:,:]
v = fh.variables['v'][t,29,:,:]
temp = fh.variables['temp'][t,29,:,:]  

# Setup map
##############################################################################
m = Basemap(projection='merc', llcrnrlat=-38.050653, urcrnrlat=-34.453367,\
        llcrnrlon=147.996456, urcrnrlon=152.457344, lat_ts=20, resolution='h')
##############################################################################

m_lons,m_lats=m(lons,lats) # the missing ingredient!

# make plot
plt.close("all")
fig = plt.figure()
fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
# draw stuff
m.drawcoastlines()
m.fillcontinents(color='black')
# plot color
m.pcolor(lons,lats,np.squeeze(temp), latlon = True ,vmin=13., vmax=25., cmap='plasma')
plt.colorbar()
# vector field
q = m.quiver(m_lons[::,::3],m_lats[::3,::3],u[:,0:165][::3,::3],v[::3,::3],color='black')
# datetime title
plt.title('Vector field test')
plt.tight_layout()

plt.show()
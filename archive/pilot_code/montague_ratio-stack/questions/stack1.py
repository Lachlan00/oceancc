# Stack Overlow NC slicing question
from netCDF4 import Dataset

# grid point lists
lat = [20, 45, 56, 67, 88, 98, 115]
lon = [32, 38, 48, 58, 87, 92, 143]

# open netCDF file
nc_file = "./sresa1b_ncar_ccsm3-example.nc"
fh = Dataset(nc_file, mode='r')

# extract variable
point_list = zip(lat,lon)
ua_list = []
for i, j in point_list:
   	ua_list.append(fh.variables['ua'][0,16,i,j])

print(ua_list)

# extract variable
ua_list = fh.variables['ua'][0,16,lat,lon]

print(ua_list)
    
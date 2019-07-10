# script to categorize and learn about the ROMS NetCDF data

# load packages
library(ncdf4) # for working with NetCDF files

# setwd
setwd("~/PhD_Large_Data/ROMS/Montague_subset")

# grab filenames
file_ls <- list.files(pattern = "\\.nc$")
# grab random filename
ncfname <- file_ls[42]

# open file
ncin <- nc_open(ncfname)

# get variables
temp <- ncvar_get(ncin, "temp")
salt <- ncvar_get(ncin, "salt")
lon <- ncvar_get(ncin, "lon_rho")
lat <- ncvar_get(ncin, "lat_rho")

# get dimensions

# close NetCDF
nc_close(ncin)




# data processes
import math
import numpy as np
from datetime import datetime
from datetime import timedelta

#######################
# Add to list quickly #
#######################
def add(lst, obj, index): return lst[:index] + [obj] + lst[index:]

###################
# Query yes or no #
###################
def yes_or_no(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply == 'y':
        return True
    if reply == 'n':
        return False
    else:
        return yes_or_no("Not a vaild response..")

#############################################
# Find min and max values in area in netCDF #
#############################################
def minmax_in_region(data, lons, lats, region):
    """
    Calculate minimum and maximum values in georeferenced array
    Give range as `range = [xmin, xmax, ymin, ymax]`
    """
    # constrain data to region
    bools = (region[0] <= lons) & (lons <= region[1]) & (region[2] <= lats) & (lats <= region[3])  
    data = data[bools]
    # get min and max
    local_min = data.min()
    local_max = data.max()

    return (local_min, local_max)

####################################################
# Convert ocean time (days since 1990) to datetime #
####################################################
def oceantime_2_dt(frame_time):
    """
    Datetime is in local timezone (but not timezone aware)
    """
    dtcon_days = frame_time
    dtcon_start = datetime(1990,1,1)
    dtcon_delta = timedelta(dtcon_days/24/60/60)
    dtcon_offset = dtcon_start + dtcon_delta

    return dtcon_offset

######################
# Harversine formula #
######################
def harversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # harversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2.)**2. + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2.)**2.
    c = 2. * math.asin(math.sqrt(a))
    km = 6371. * c # radius of earth
    return km
    
##################
# Make study box #
##################
def boxmaker(lon_orig, lat_orig, km):
    """
    Calculate points directly north, south, east and 
    west a certain distance from given coordinates
    Won't handle a c
    """
    # convert decimal degrees to radians
    lon_orig, lat_orig = map(math.radians, [lon_orig, lat_orig])
    # reverse harversine formula
    c = km / 6371.
    a = math.sin(c/2.)**2.
    dlat = 2. * math.asin(math.sqrt(a))
    dlon = 2. * math.asin(math.sqrt(a/(math.cos(lat_orig)**2.)))
    # convert back to decimal degrees 
    lon_orig, lat_orig, dlat, dlon = map(math.degrees, [lon_orig, lat_orig, dlat, dlon])
    # find coordinates
    north = lat_orig + dlat
    south = lat_orig - dlat
    east = lon_orig + dlon
    west = lon_orig - dlon
    # correct over the 0-360 degree line
    if west > 360:
        west = west - 360
    if east > 360:
        east = east - 360
    # round to 6 decimal places
    region = [west, east, south, north]
    region = [round(x,6) for x in region]

    # export region
    return region

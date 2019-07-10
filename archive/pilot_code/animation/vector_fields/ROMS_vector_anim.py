"""
Produce animation of ROMS data 
"""
print('----------------------------')
print('NetCDF animation builder v-2.0')
print('----------------------------')
print('----Program Start----')

print('loading packages...')
#Packages
from netCDF4 import Dataset # reads netCDF file
import numpy as np # manipulates arrays
import pandas as pd # for dataframes and reading track csv data
import matplotlib.pyplot as plt # for plotting map
from mpl_toolkits.basemap import Basemap # basemap tools
from datetime import datetime, timedelta #for working with datetimes
import moviepy.editor as mpy # creates animation
from moviepy.video.io.bindings import mplfig_to_npimage # converts map to numpy array
from matplotlib.backends.backend_agg import FigureCanvasAgg # draws canvas so that map can be converted
from os import listdir
from os.path import isfile, join

# define animation buiding functions
##########__FUNCTIONS__##########

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

def plot_vectors(t):
    u = fh.variables['u'][t,29,:,:]
    v = fh.variables['v'][t,29,:,:]
    q = m.quiver(m_lons[::3,::3],m_lats[::3,::3],u[:,0:165][::3,::3],v[::3,::3],color='black')

def make_frame_temp(frame_idx):
    plt.close("all")
    """
    Make animation of just temperature
    """
    # set start frame
    start_frame = 0

    # make frame__idx an integer to avoid slicing errors
    frame_idx = int(frame_idx)

    # import salt at timestamp
    temp = fh.variables['temp'][start_frame + frame_idx,29,:,:] 

    # get 'frame_time'
    frame_time = grab_sst_time(start_frame + frame_idx)
   
    # map setup
    fig = plt.figure()
    fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # Setup the map
    # m = Basemap(projection='merc', llcrnrlat=-38.050653, urcrnrlat=-34.453367,\
    #         llcrnrlon=147.996456, urcrnrlon=152.457344, lat_ts=20, resolution='h') # full range
    # m = Basemap(projection='merc', llcrnrlat=-36.701671, urcrnrlat=-35.802349,\
    #         llcrnrlon=149.669301, urcrnrlon=150.784499, lat_ts=20, resolution='h') # 100km^2
    # m = Basemap(projection='merc', llcrnrlat=-36.476840, urcrnrlat=-36.027180,\
    #        llcrnrlon=149.948101, urcrnrlon=150.505699, lat_ts=20, resolution='h') # 50km^2
    # draw stuff
    m.drawcoastlines() # comment out when using shapefile
    m.fillcontinents(color='black')
    # use shapefile coastline
    # m.readshapefile('/Users/lachlanphillips/Dropbox/PhD/Data/GIS/shp_files/NSW-coastline/Edited/polygon/NSW-coastline_WGS84', 'NSW-coastline_WGS84')
    # plot salt
    cs = m.pcolor(lons,lats,np.squeeze(temp), latlon = True ,vmin=temp_min, vmax=temp_max, cmap='plasma')
    # plot colourbar
    plt.colorbar()
    # vectors
    plot_vectors(frame_idx)
    # datetime title
    plt.title('Regional - Temperature (Celcius)\n' + frame_time.strftime("%Y-%m-%d %H:%M:%S") + ' | ' + str(fname) + '_idx: ' + str(frame_idx).zfill(2))
    # stop axis from being cropped
    plt.tight_layout() 
    
    #convert to array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    frame = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return frame

def make_frame_salt(frame_idx):
    plt.close("all")
    """
    Make animation of just salinity
    """
    # set start frame
    start_frame = 0

    # make frame__idx an integer to avoid slicing errors
    frame_idx = int(frame_idx)

    # import salt at timestamp
    salt = fh.variables['salt'][start_frame + frame_idx,29,:,:] 

    # get 'frame_time'
    frame_time = grab_sst_time(start_frame + frame_idx)
   
    # map setup
    fig = plt.figure()
    fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # Setup the map
    # m = Basemap(projection='merc', llcrnrlat=-38.050653, urcrnrlat=-34.453367,\
    #         llcrnrlon=147.996456, urcrnrlon=152.457344, lat_ts=20, resolution='h') # full range
    # m = Basemap(projection='merc', llcrnrlat=-36.701671, urcrnrlat=-35.802349,\
    #         llcrnrlon=149.669301, urcrnrlon=150.784499, lat_ts=20, resolution='h') # 100km^2
    # m = Basemap(projection='merc', llcrnrlat=-36.476840, urcrnrlat=-36.027180,\
    #         llcrnrlon=149.948101, urcrnrlon=150.505699, lat_ts=20, resolution='h') # 50km^2
    # draw stuff
    m.drawcoastlines() # comment out when using shapefile
    m.fillcontinents(color='black')
    # # use shapefile coastline
    # m.readshapefile('/Users/lachlanphillips/Dropbox/PhD/Data/GIS/shp_files/NSW-coastline/Edited/polygon/NSW-coastline_WGS84', 'NSW-coastline_WGS84')
    # plot salt
    cs = m.pcolor(lons,lats,np.squeeze(salt), latlon = True ,vmin=salt_min, vmax=salt_max, cmap='viridis')
    # plot colourbar
    plt.colorbar()
    # vectors
    plot_vectors(frame_idx)
    # datetime title
    plt.title('Regional - Salinity (PSU)\n' + frame_time.strftime("%Y-%m-%d %H:%M:%S") + ' | ' + str(fname) + '_idx: ' + str(frame_idx).zfill(2))
    # stop axis from being cropped
    plt.tight_layout() 
    
    #convert to array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    frame = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return frame

# MAIN PROGRAM
# set colour scale variables
temp_min = 13
temp_max = 25 
salt_min = 35.3
salt_max = 35.7

# get list of files in data directory
in_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset"
file_ls = [f for f in listdir(in_directory) if isfile(join(in_directory, f))]
file_ls = list(filter(lambda x:'naroom_avg' in x, file_ls))
file_ls = sorted(file_ls)

# set output directory
out_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/gifs"

##############################################################################
m = Basemap(projection='merc', llcrnrlat=-38.050653, urcrnrlat=-34.453367,\
        llcrnrlon=147.996456, urcrnrlon=152.457344, lat_ts=20, resolution='h')
##############################################################################

# set range
start = 276
end = 277
# make animation for each file (will take over 10 hours)
for i in range(start, end):
    # import file
    nc_file = in_directory + '/' + file_ls[i]
    fh = Dataset(nc_file, mode='r')
    print('Building animations with file: '+file_ls[i]+' | '+str(i+1)+' of '+ str(len(file_ls)))
    fname = str(file_ls[i])[11:16]

    # set gif names
    out_temp = out_directory+'/full_run/vectors/temp/'+'temp'+str(i+1).zfill(3)+'.gif'
    out_salt = out_directory+'/full_run/vectors/salt/'+'salt'+str(i+1).zfill(3)+'.gif'

    # extract data
    lats = fh.variables['lat_rho'][:] 
    lons = fh.variables['lon_rho'][:]
    time = fh.variables['ocean_time'][:]
    m_lons,m_lats=m(lons,lats) # for vector fields

    # make animations
    frame_count = len(time)
    # salt
    animation = mpy.VideoClip(make_frame_salt, duration=frame_count)
    animation.write_gif(out_salt, fps=1)
    # temp
    animation = mpy.VideoClip(make_frame_temp, duration=frame_count)
    animation.write_gif(out_temp, fps=1)
    # close file
    fh.close()

print('-----Program End-----')

print('')
print('´´´´´´´´´´´´´´´´´´´´´´¶¶¶¶¶¶¶¶¶……..')
print('´´´´´´´´´´´´´´´´´´´´¶¶´´´´´´´´´´¶¶……')
print('´´´´´´¶¶¶¶¶´´´´´´´¶¶´´´´´´´´´´´´´´¶¶……….')
print('´´´´´¶´´´´´¶´´´´¶¶´´´´´¶¶´´´´¶¶´´´´´¶¶…………..')
print('´´´´´¶´´´´´¶´´´¶¶´´´´´´¶¶´´´´¶¶´´´´´´´¶¶…..')
print('´´´´´¶´´´´¶´´¶¶´´´´´´´´¶¶´´´´¶¶´´´´´´´´¶¶…..')
print('´´´´´´¶´´´¶´´´¶´´´´´´´´´´´´´´´´´´´´´´´´´¶¶….')
print('´´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´´´´´´´´´´´´´´´´´´´´¶¶….')
print('´´´¶´´´´´´´´´´´´¶´¶¶´´´´´´´´´´´´´¶¶´´´´´¶¶….')
print('´´¶¶´´´´´´´´´´´´¶´´¶¶´´´´´´´´´´´´¶¶´´´´´¶¶….')
print('´¶¶´´´¶¶¶¶¶¶¶¶¶¶¶´´´´¶¶´´´´´´´´¶¶´´´´´´´¶¶…')
print('´¶´´´´´´´´´´´´´´´¶´´´´´¶¶¶¶¶¶¶´´´´´´´´´¶¶….')
print('´¶¶´´´´´´´´´´´´´´¶´´´´´´´´´´´´´´´´´´´´¶¶…..')
print('´´¶´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´´´´´´´´´´´´´´´¶¶….')
print('´´¶¶´´´´´´´´´´´¶´´¶¶´´´´´´´´´´´´´´´´¶¶….')
print('´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´¶¶´´´´´´´´´´´´¶¶…..')
print('´´´´´´´´´´´´´´´´´´´´´´´¶¶¶¶¶¶¶¶¶¶¶…….)')
print('')
print('Tip: To increase fame speed, use the following commands to use imagemagick from terminal:')
print('convert -delay 25x100 gif_name.gif gif_name.gif')
print('NetCDF animation builder v-2.0 does not currently support subplots.')
print('For now use "https://ezgif.com/combine" to join the gifs.')

# Additional function to make both plots on one GIF. Needs to be debugged. 
def make_frame_dual(frame_idx):
    """
    Make animation of just salinity
    NOTE: Until I work out the solution for plotting colorbars using `ax` 
    variables this website is a great alternative solution to combine the gifs: 
    https://ezgif.com/combine
    """
    # set start frame
    start_frame = 0

    # make frame__idx an integer to avoid slicing errors
    frame_idx = int(frame_idx)

    # import salt and temp at timestamp
    salt = fh.variables['salt'][start_frame + frame_idx,29,:,:] 
    temp = fh.variables['temp'][start_frame + frame_idx,29,:,:]

    # get 'frame_time'
    frame_time = grab_sst_time(start_frame + frame_idx)

    # set up figure
    fig = plt.figure()
    fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)

    # Temperature figure
    plt.subplot(1, 2, 1)
    # fig1.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # Setup the map
    m = Basemap(projection='merc', llcrnrlat=-38.050653, urcrnrlat=-34.453367,\
            llcrnrlon=147.996456, urcrnrlon=152.457344, lat_ts=20, resolution='h')
    # draw stuff
    m.drawcoastlines()
    m.fillcontinents(color='black')
    # plot salt
    cs = m.pcolor(lons,lats,np.squeeze(temp), latlon = True ,vmin=temp_min, vmax=temp_max, cmap='plasma')
    # plot colourbar
    plt.colorbar()
    # datetime title
    plt.title('Regional - Temperature (Celcius)\n' + frame_time.strftime("%Y-%m-%d %H:%M:%S") + ' | ' + str(fname) + '_idx: ' + str(frame_idx))
   
    # Salinity figure
    plt.subplot(1, 2, 2)
    # fig2.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # Setup the map
    m = Basemap(projection='merc', llcrnrlat=-38.050653, urcrnrlat=-34.453367,\
            llcrnrlon=147.996456, urcrnrlon=152.457344, lat_ts=20, resolution='h')
    # draw stuff
    m.drawcoastlines()
    m.fillcontinents(color='black')
    # plot salt
    cs = m.pcolor(lons,lats,np.squeeze(salt), latlon = True ,vmin=salt_min, vmax=salt_max, cmap='viridis')
    # plot colourbar
    plt.colorbar()
    # datetime title
    plt.title('Regional - Salinity (PSU)\n' + frame_time.strftime("%Y-%m-%d %H:%M:%S") + ' | ' + str(fname) + '_idx: ' + str(frame_idx))
    
    # make layout nice
    plt.tight_layout()
    
    # convert to array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    frame = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return frame

# Check the sensitivity of the module to training data boundaries
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
from progressbar import ProgressBar
import cmocean
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
import seaborn as sns
import joypy
from datetime import datetime
from netCDF4 import Dataset
from colormap import rgb2hex

from train import *
from classify import *
from data_processes import *
from data_visulisation import *

# Hack to fix missing PROJ4 env var for basemap
import os
import conda
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap

# box modification function
def boxMod(sourcebox, increment, iterations, NS='NA', EW='NA'):
    """
    Method not ideal. Need to update so boxes diverge.
    Or make modification custom.
    """
    # determine direction
    if NS == 'N':
        v = 1
    elif NS == 'S':
        v = -1
    else:
        v = 0
    if EW == 'E':
        u = 1
    elif EW == 'W':
        u = -1
    else:
        u = 0

    box_ls = [sourcebox]*iterations
    return [[box[0]+(increase*u), box[1]+(increase*u), box[2]+(increase*v), box[3]+(increase*v)] for box, increase in zip(box_ls, np.arange(0, increment*iterations, increment))]

# sensitivty function
def sensitivity_obtainData(name, input_directory, output_directory, sourcebox_modA, sourcebox_modB, increment, iterations=10):
    print('\nGetting data to test model sensitivty...')
    output_directory = output_directory + '/' + name + '/'
    # check if training data already exists
    if os.path.exists(output_directory+'sensitivity1_training_data.csv'):
        if yes_or_no('Previous sensitivty data has been detetcted. Would you like to produce a new training dataset?'):
            print('Previous data will be overwirtten.')
        else:
            print('Using previously extracted dataset.')
            return

    # modify source box positions
    boxAmod_ls = sourcebox_modA
    boxBmod_ls = sourcebox_modB

    # extract_data
    for iteration in range(0, iterations):
        print('\nIteration '+str(iteration+1)+' of '+str(iterations))
        train_CARS(input_directory, output_directory+'sensitivity'+str(iteration+1)+'_', boxAmod_ls[iteration], boxBmod_ls[iteration])

def sensitivity_runModels(name, ROMS_directory, training_directory, region, depthmax=1e10, iterations=10):
    print('\nRunning model sensitivity checks..')
    training_directory = training_directory + '/' + name + '/'
    # get training data file directories
    file_ls = [f for f in listdir(training_directory) if isfile(join(training_directory, f))]
    file_ls = list(filter(lambda x:'training_data.csv' in x, file_ls))
    file_ls.sort(key=natural_keys)
    # reset requested iterations if greater than data avaliable
    if len(file_ls) < iterations:
        print(str(iterations)+' iterations requested but only '+str(len(file_ls))+' data files are avliable.')
        iterations = len(file_ls)
        print(str(iterations)+' model iterations will be run.')

    # make the models
    print('\nBuilding and cross validating lostic regression models..')
    lr_models = [None]*iterations
    pbar = ProgressBar(max_value=len(file_ls))
    pbar.update(0)
    for i in range(0, iterations):
        lr_models[i] = current_model(training_directory + file_ls[i], verbose=False)
        pbar.update(i+1)

    # extract the count data
    print('\nRunning the models..')
    for i in range(0, iterations):
        print('\nModel run '+str(i+1)+' of '+str(iterations))
        df = analyse_region_counts(ROMS_directory, lr_models[i], region, depthmax=1e10)
        df.to_csv(training_directory + 'sensitivity'+str(i+1)+'_countData.csv', index=False)

def sensitivity_analysisMap(name, CARS_directory, sourcebox_modA, sourcebox_modB, region, increment, iterations=10, save=False):
    # modify source box positions
    # imporve resolution (note)
    boxAmod = np.asarray(sourcebox_modA)
    boxBmod = np.asarray(sourcebox_modB)
    boxAmod_ravel = boxAmod.ravel()
    boxBmod_ravel = boxBmod.ravel()

    # get CARS temperasture data
    file_path = CARS_directory + 'temperature_cars2009a.nc'
    fh = Dataset(file_path, mode='r')
    T_mean = fh.variables['mean'][0,:,:]
    # get lons and lats
    lons = fh.variables['lon'][:]
    lats = fh.variables['lat'][:]
    # make into 2d arrays
    lons = np.tile(lons, 331)
    lons = np.reshape(lons,(331,721))
    lats = np.repeat(lats,721)
    lats = np.reshape(lats,(331,721))

    # make map
    all_lats = np.asarray([[boxA[2:4], boxB[2:4]] for boxA, boxB in zip(boxAmod, boxBmod)]).ravel()
    all_lons = np.asarray([[boxA[0:2], boxB[0:2]] for boxA, boxB in zip(boxAmod, boxBmod)]).ravel()
    all_lats = np.asarray(list(all_lats) + region[2:4])
    all_lons = np.asarray(list(all_lons) + region[0:2])
    map_region = [min(all_lons)-5, max(all_lons)+5, min(all_lats)-5, max(all_lats)+5]
    m = Basemap(projection='merc', llcrnrlat=min(all_lats)-5, urcrnrlat=max(all_lats)+5,\
        llcrnrlon=min(all_lons)-5, urcrnrlon=max(all_lons)+5, lat_ts=20, resolution='h')
  
    # make fig  
    fig = plt.figure(figsize=(5,15))
    fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # draw stuff
    m.drawcoastlines(color='black', linewidth=0.7)
    m.fillcontinents(color='#A0A0A0')

    # colour map of tmeperature
    # get max temps
    vmin, vmax = minmax_in_region(T_mean, lons, lats, map_region)
    cs = m.pcolor(lons, lats, np.squeeze(T_mean), latlon=True, vmin=vmin, 
        vmax=vmax, cmap=cmocean.cm.thermal)

    # add grid
    parallels = np.arange(-81.,0,5.)
    m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    meridians = np.arange(10.,351.,5.)
    m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')

    # make verts
    vertsA = np.asarray([[m(x[0], x[2]),m(x[0], x[3]),m(x[1], x[3]),m(x[1], x[2])] for x in boxAmod])
    vertsB = np.asarray([[m(x[0], x[2]),m(x[0], x[3]),m(x[1], x[3]),m(x[1], x[2])] for x in boxBmod])
    # colour scale
    z = np.arange(1, len(boxAmod)+1, 1)
    # make ploygon collections
    collA = PolyCollection(verts=vertsA, array=z, cmap=cmocean.cm.haline, facecolor='none', edgecolor='black')
    collB = PolyCollection(verts=vertsB, array=z, cmap=cmocean.cm.haline, facecolor='none', edgecolor='black')
    # add to plot
    plt.gca().add_collection(collA)
    plt.gca().add_collection(collB)

    # add study region
    plt.gca().add_patch(make_polygon(region, m))

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)

    # stop axis from being cropped
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.05)

    plt.show()

    if save:
        print('\nSaving plot..')
        fig.savefig('./data/sensitivity/'+name+'/plots/sensitivity_map.png')

    # show plot
    print('Close plot to continue..')
    plt.show()
    plt.close("all")

def sensitivity_analysisJoy(name, data_directory, iterations=10, save=False):
    data_directory = data_directory + name + '/' 
    # get training data file directories
    file_ls = [f for f in listdir(data_directory) if isfile(join(data_directory, f))]
    file_ls = list(filter(lambda x:'countData.csv' in x, file_ls))
    file_ls.sort(key=natural_keys)

    letters = 'abcdefghijklmnopqrstuvwxyz'

    # reset requested iterations if greater than data avaliable
    if (len(file_ls) < iterations):
        print(str(iterations)+' iterations requested but only '+str(len(file_ls))+' data files are avliable.')
        iterations = len(file_ls)
        print(str(iterations)+' model iterations will be run.')

    # load each dataframe of count data
    data = [pd.read_csv(data_directory+file, parse_dates=['dt'],
                        infer_datetime_format=True) for file in file_ls]

    # get mean seasonality
    data = [mean_ratioA(df) for df in data]
    index = data[0]['index']
    index = [datetime(2000, dt.month, 1) for dt in index]
    index = set(index)
    # smooth
    for df in data:
        df['ratioA_smooth'] = savgol_filter(df['ratioA'], 35, 3)

    # Create the data
    x = [list(df['ratioA_smooth']) for df in data]
    x = list(zip(*x))
    x = np.asarray([item for sublist in x for item in sublist])
    g = np.tile(list([str(n) for n in letters[0:iterations]]) , len(data[1]['ratioA']))
    df = pd.DataFrame(dict(x=x, g=g))

    # joyploy
    x_range = list(range(366))
    fig, axes = joypy.joyplot(df, by="g", colormap=cmocean.cm.haline, kind='values', 
        x_range=x_range, labels=['']*iterations, overlap=3, grid='y', alpha=1)

    # edit axis
    X = plt.gca().xaxis
    X.set_major_locator(mdates.MonthLocator())
    X.set_major_formatter(mdates.DateFormatter('%b'))
    axes[-1].set_xlim(datetime(2000, 1, 1),datetime(2000, 12, 31))

    if save:
        print('\nSaving plot..')
        fig.savefig('./data/sensitivity/'+name+'/plots/sensitivity_joy.png')

    # show plot
    print('Close plot to continue..')
    plt.show()
    plt.close("all")

def sensitivity_analysisLine(name, data_directory, iterations, save=False):
    data_directory = data_directory + name + '/' 
    # get training data file directories
    file_ls = [f for f in listdir(data_directory) if isfile(join(data_directory, f))]
    file_ls = list(filter(lambda x:'countData.csv' in x, file_ls))
    file_ls.sort(key=natural_keys)

    # reset requested iterations if greater than data avaliable
    if (len(file_ls) < iterations):
        print(str(iterations)+' iterations requested but only '+str(len(file_ls))+' data files are avliable.')
        iterations = len(file_ls)
        print(str(iterations)+' model iterations will be run.')

    # load each dataframe of count data
    data = [pd.read_csv(data_directory+file, parse_dates=['dt'],
                        infer_datetime_format=True) for file in file_ls]

    # get mean seasonality
    data = [mean_ratioA(df) for df in data]
    index = data[0]['index']
    index = [datetime(2000, dt.month, 1) for dt in index]
    index = set(index)
    # smooth
    for df in data:
        df['ratioA_smooth'] = savgol_filter(df['ratioA'], 35, 3)

    # make plot
    fig, ax = plt.subplots(figsize=(12, 3))
    plt.grid(ls='dashed', alpha=0.7)
    # add lines
    cmaps = cmocean2rgb(cmocean.cm.haline, len(data))
    cmaps = [x[1] for x in cmaps]
    cmaps = [eval(x.strip('rgb')) for x in cmaps]
    cmaps = [tuple(int(x) for x in cmap) for cmap in cmaps]
    cmaps = [rgb2hex(*cmap) for cmap in cmaps]

    for df, cmap in zip(data, cmaps):
        plt.plot(df['index'], df['ratioA_smooth'], color=cmap, alpha=0.6)

    if save:
        print('\nSaving plot..')
        fig.savefig('./data/sensitivity/'+name+'/plots/sensitivity_joy.png')

    # show plot
    print('Close plot to continue..')
    plt.show()
    plt.close("all")






















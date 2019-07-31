# make plots
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.dates as mdates
import cmocean # oceanogrpahy colorscales - https://matplotlib.org/cmocean/
from datetime import datetime
from datetime import timedelta
import PIL
import numpy as np
from os import listdir
from os.path import isfile, join
from netCDF4 import Dataset

# Hack to fix missing PROJ4 env var for basemap
import os
import conda
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap

# local modules
from data_processes import *

################
# plot polygon #
################
def make_polygon(region, m):
    x1,y1 = m(region[0],region[2]) 
    x2,y2 = m(region[0],region[3]) 
    x3,y3 = m(region[1],region[3]) 
    x4,y4 = m(region[1],region[2])
    p = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)], facecolor='none',
                edgecolor='#ff0000',linewidth=2,ls='dashed', zorder=10)

    return(p)

###############################################
# Check box positions relative to ROMS extent #
###############################################
def check_boxROMS(box_ls, ROMS_directory, depthmax=1e10, save=False, out_fn=''):
    print('\nChecking analysis regions..')
    file_ls = [f for f in listdir(ROMS_directory) if isfile(join(ROMS_directory, f))]
    file_ls = list(filter(lambda x:'.nc' in x, file_ls))
    file_ls = sorted(file_ls)
    # obtain bathymetry data
    nc_file = ROMS_directory + '/' + file_ls[0]
    fh = Dataset(nc_file, mode='r')
    lats = fh.variables['lat_rho'][:] 
    lons = fh.variables['lon_rho'][:]
    bath = fh.variables['h'][:]

    # Check balance
    print('Study zone balance (grid cell count)..')
    for region, i in zip(box_ls, range(1,len(box_ls)+1)):
        cells = count_points(lons, lats, region, bath, depthmax)
        print('Box '+str(i)+': '+str(cells))

    # convert bath to km
    bath = bath/1000

    # setup map
    m = Basemap(projection='merc', llcrnrlat=lats.min()-1, urcrnrlat=lats.max()+1,\
        llcrnrlon=lons.min()-1, urcrnrlon=lons.max()+1, lat_ts=20, resolution='h')

    # make plot
    fig = plt.figure(figsize=(8,8))
    plt.tight_layout()
    plt.title('Analysis Regions', size=15)
    # draw stuff
    m.drawcoastlines(color='black', linewidth=0.7)
    m.fillcontinents(color='#A0A0A0')
    # add grid
    parallels = np.arange(-81.,0,2.)
    m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    meridians = np.arange(10.,351.,2.)
    m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    # add bathymetry extent
    cs = m.pcolor(lons, lats, np.squeeze(bath), latlon=True ,vmin=0, vmax=5, cmap=cmocean.cm.deep)
    cbar = plt.colorbar(ticks=np.arange(6))
    CS = m.contour(lons, lats, np.squeeze(bath), [depthmax/1000], colors='k', latlon=True, linestyles='dashed', alpha=0.9)
    cbar.ax.invert_yaxis()
    cbar.ax.set_ylabel('Depth (km)', rotation=270, labelpad=16, size=12)

     # add polygons
    for i in range(0, len(box_ls)): 
         plt.gca().add_patch(make_polygon(box_ls[i], m))

    # save plot
    if save:
        print('\nSaving plot..')
        fig.savefig(out_fn)

    # show plot
    print('Close plot to continue..')
    plt.show()
    plt.close("all")

###########################################
# Analyse seasonal variability of current #
###########################################
def seasonal_change_analysis(df, title, out_fn):
    # calculate ratios
    df['ratioA'] = [A/(A+B) for A, B in zip(df['countA'], df['countB'])]
    df['ratioB'] = [B/(A+B) for A, B in zip(df['countA'], df['countB'])]
    # add date components
    df['day'] = [x.day for x in df['dt']]
    df['month'] = [x.month for x in df['dt']]
    df['year'] = [x.year for x in df['dt']]

    # Seasonal data
    df = df[df.year != 2016] # drop 2016 (incomplete)
    # calc yearly means
    df_std = df.groupby(['month', 'day']).std().reset_index()
    df_mean = df.groupby(['month', 'day']).mean().reset_index()
    # add seasonal first half vs second half
    df1 = df.set_index(['dt'])
    df2 = df1
    df1 = df1.loc['1994-1-1':'1999-12-31']
    df2 = df2.loc['2010-1-1':'2015-12-31']
    df1 = df1.groupby(['month', 'day']).mean().reset_index()
    df2 = df2.groupby(['month', 'day']).mean().reset_index()

    # build index
    index = df_mean.index
    base = datetime(2000, 1, 1, 0, 0, 0)
    index = [base + timedelta(int(x)) for x in index]

    # make plot
    fig, ax = plt.subplots(figsize=(20, 3))
    plt.grid(ls='dashed', alpha=0.7)
    ax.stackplot(index, list(df_mean['ratioA']), color='#606060')
    # make bands
    ub = df_mean['ratioA'] + df_std['ratioA']
    lb = df_mean['ratioA'] - df_std['ratioA']
    ub = [1 if x >= 1 else x for x in ub]
    lb = [0 if x <= 0 else x for x in lb]
    # shade between bands
    # plt.fill_between(index, lb, ub, alpha=0.25, color='#4682B4')
    # add first and last 5 year lines
    plt.plot(index, df1.ratioA, '--', color='b', alpha=0.6)
    plt.plot(index, df2.ratioA, '--', color='r', alpha=0.6)
    # title
    plt.ylabel('Classification Ratio', labelpad=16, size = 14)
    plt.title(title, size=15)
    # edit axis
    X = plt.gca().xaxis
    X.set_major_locator(mdates.MonthLocator())
    X.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_xlim(datetime(2000, 1, 1),datetime(2000, 12, 31))

    # report data
    print('\nAnalysis report for "'+title+'"')
    print('First 5 years: '+str(round(df1.ratioA.mean(),4))+' ('+str(round(df1.ratioA.mean()*12,4))+' months)')
    print('Last 5 years: '+str(round(df2.ratioA.mean(),4))+' ('+str(round(df2.ratioA.mean()*12,4))+' months)')
    print('Difference: '+str(round(df2.ratioA.mean() - df1.ratioA.mean(),4))+' ('+
        str(round((df2.ratioA.mean() - df1.ratioA.mean())*12,4))+' months)')

    # save plot
    print('\nMaking plot..')
    plt.show()
    fig.savefig(out_fn)
    plt.close("all")

###########################################################
# Analyse change in the monthly mean of current influence #
###########################################################
def monthly_change_analysis(df, title, out_fn):
    # calculate ratios
    df['ratioA'] = [A/(A+B) for A, B in zip(df['countA'], df['countB'])]
    df['ratioB'] = [B/(A+B) for A, B in zip(df['countA'], df['countB'])]
    # add date components
    df['day'] = [x.day for x in df['dt']]
    df['month'] = [x.month for x in df['dt']]
    df['year'] = [x.year for x in df['dt']]


#########################
# Merge images together #
#########################
def img_join(output_name, img_ls, direction='vertical'):
    # open images
    images = [PIL.Image.open(i) for i in img_ls]
    # get smallest image and resizse others
    min_shape = sorted([(np.sum(i.size), i.size) for i in images])[0][1]
    imgs_comb = np.hstack((np.asarray( i.resize(min_shape)) for i in images))

    if direction == 'vertical':
        imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in images))
        imgs_comb = PIL.Image.fromarray(imgs_comb)
        imgs_comb.save(output_name)
    elif direction == 'horizontal':
        imgs_comb = PIL.Image.fromarray(imgs_comb)
        imgs_comb.save(output_name)    
    else:
        print('Error: invalid direction string.')

    print('\nImages joined - saved to: '+output_name)

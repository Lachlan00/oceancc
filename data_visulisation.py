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
from scipy import stats
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import itertools,operator

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

# DEBUG
import pdb

def cmocean2rgb(cmap, n):
    """
    Function to convert cmocean colormaps to be compatible with plotly
    """
    h = 1.0/(n-1)
    pl_colorscale = []
    for k in range(n):
        C = np.array(cmap(k*h)[:3])*255
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
    return pl_colorscale

def clamp(x): 
  return max(0, min(x, 255))

################
# plot polygon #
################
def make_polygon(region, m, edgecolor='#e60000'):
    x1,y1 = m(region[0],region[2]) 
    x2,y2 = m(region[0],region[3]) 
    x3,y3 = m(region[1],region[3]) 
    x4,y4 = m(region[1],region[2])
    p = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)], facecolor='none',
                edgecolor=edgecolor,linewidth=2,ls='dashed', zorder=10)

    return(p)

###############################################
# Check box positions relative to ROMS extent #
###############################################
def check_boxROMS(box_ls, ROMS_directory, depthmax=1e10, save=False, out_fn='FIG.png', 
    labels=None, title=None, zoom2box=False):
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
    if zoom2box & (len(box_ls) <= 1):
        m = Basemap(projection='merc', llcrnrlat=box_ls[0][2] - .2, urcrnrlat=box_ls[0][3] + .2,\
            llcrnrlon=box_ls[0][0] - .2, urcrnrlon=box_ls[0][1] + .2, lat_ts=20, resolution='h')
    else:
        m = Basemap(projection='merc', llcrnrlat=lats.min()-1, urcrnrlat=lats.max()+1,\
            llcrnrlon=lons.min()-1, urcrnrlon=lons.max()+1, lat_ts=20, resolution='h')

    # make plot
    fig = plt.figure(figsize=(8,8))
    plt.tight_layout()
    # set title
    if not title is None:
        plt.title(title, size=15)
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
    cbar = plt.colorbar(ticks=np.arange(6), pad=0.05, shrink=0.95)
    CS = m.contour(lons, lats, np.squeeze(bath), [depthmax/1000], colors='k', latlon=True, linestyles='dashed', alpha=0.9)
    cbar.ax.invert_yaxis()
    cbar.ax.set_ylabel('Depth (km)', rotation=270, labelpad=16, size=12)

     # add polygons
    for i in range(0, len(box_ls)): 
         plt.gca().add_patch(make_polygon(box_ls[i], m))

    # add labels
    if labels is not None:
        for region, label in zip(box_ls, labels):
            x, y = m(region[0],region[3]-(region[3]-region[2])/2)
            x = x - 23000
            txt = plt.text(x, y, label+' '+u'\u25B6', size=8.5, ha='right', alpha=0.8)
            txt.set_bbox(dict(facecolor='#e6e6e6', alpha=0.5, edgecolor='#e6e6e6'))

    # save plots
    if save:
        print('\nSaving plot..')
        fig.savefig(out_fn, dpi=300)

    # show plot
    print('Close plot to continue..')
    plt.show()
    plt.close("all")

##############
# Map inset #
##############
def map_inset(ROMS_directory, save=True, out_fn='map_inset.png', eac_panel_inset=False):
    file_ls = [f for f in listdir(ROMS_directory) if isfile(join(ROMS_directory, f))]
    file_ls = list(filter(lambda x:'.nc' in x, file_ls))
    file_ls = sorted(file_ls)
    # obtain bathymetry data
    nc_file = ROMS_directory + '/' + file_ls[0]
    fh = Dataset(nc_file, mode='r')
    lats = fh.variables['lat_rho'][:] 
    lons = fh.variables['lon_rho'][:]
    # setup map
    m = Basemap(projection='merc', llcrnrlat=lats.min()-6, urcrnrlat=lats.max()+26,\
        llcrnrlon=lons.min()-37, urcrnrlon=lons.max()+12, lat_ts=20, resolution='h')

    # make plot
    fig = plt.figure(figsize=(2,2))
    #splt.tight_layout()
    # draw stuff
    m.drawcoastlines(color='black', linewidth=0.7)
    m.fillcontinents(color='#A0A0A0')
    # add grid
    parallels = np.arange(-81.,0,15.)
    m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    meridians = np.arange(10.,351.,20.)
    m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    # add polygon
    if eac_panel_inset:
        plt.gca().add_patch(make_polygon([lons.min()-6,lons.max()+10,lats.min()-5,lats.max()+10], m))
    else:
        plt.gca().add_patch(make_polygon([lons.min()-1,lons.max()+1,lats.min()-1,lats.max()+1], m))

    plt.gcf().subplots_adjust(left=0.25)
    # save plots
    if save:
        print('\nSaving plot..')
        fig.savefig(out_fn, dpi=300, transparent=True)

    # show plot
    print('Close plot to continue..')
    plt.show()
    plt.close("all")

##################
# EAC panel base #
##################
def eac_panel(ROMS_directory, save=True, out_fn='map_inset.png', 
    SST_fn='/Volumes/LP_MstrData/master-data/ocean/NOAA/sst.day.mean.2000.nc'):
    file_ls = [f for f in listdir(ROMS_directory) if isfile(join(ROMS_directory, f))]
    file_ls = list(filter(lambda x:'.nc' in x, file_ls))
    file_ls = sorted(file_ls)
    # obtain bathymetry data
    nc_file = ROMS_directory + '/' + file_ls[0]
    fh = Dataset(nc_file, mode='r')
    lats = fh.variables['lat_rho'][:] 
    lons = fh.variables['lon_rho'][:]
    # get SST
    fh = Dataset(SST_fn, mode='r')
    sst = fh.variables['sst'][345,:,:] 
    sst_lat = fh.variables['lat'][:]
    sst_lon = fh.variables['lon'][:]
    # tile the arrays to have coords for each grid cell
    sst_lat = np.asarray([np.tile(lat, sst.shape[1]) for lat in sst_lat])
    sst_lon = np.asarray(([sst_lon]*sst.shape[0]))
    # setup map
    m = Basemap(projection='merc', llcrnrlat=lats.min()-5, urcrnrlat=lats.max()+10,\
        llcrnrlon=lons.min()-6, urcrnrlon=lons.max()+10, lat_ts=20, resolution='h')

    # make plot
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    #plt.tight_layout()
    # draw stuff
    m.drawcoastlines(color='black', linewidth=0.7)
    m.fillcontinents(color='#A0A0A0')
    # add grid
    parallels = np.arange(-80.,0,8.)
    m.drawparallels(parallels,labels=[False,True,False,True], linewidth=1, dashes=[3,3], color='#707070')
    meridians = np.arange(10.,351.,5.)
    m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    # add sst
    cs = m.pcolor(sst_lon, sst_lat, np.squeeze(sst), latlon=True ,vmin=12, vmax=28, cmap=cmocean.cm.thermal)
    #cbaxes = fig.add_axes([0.165, 0.11, 0.045, .7703])  # Position for the colorbar [left, bottom, width, height]
    cbar = plt.colorbar(pad=0.05, shrink=0.915, ax=[ax], location='left')
    cbar.ax.set_ylabel('Sea Surface Temperature (°C)', rotation=90, labelpad=16, size=12)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    # add polygon
    #plt.gca().add_patch(make_polygon([lons.min()-1,lons.max()+1,lats.min()-1,lats.max()+1], m))
    # save plots
    if save:
        print('\nSaving plot..')
        fig.savefig(out_fn, dpi=300, transparent=True)

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

    # smooth the trend lines
    # duplicate signal so trend is smoothed properly across end and start cyle
    smooth_first = list(df1['ratioA'])*3
    smooth_last = list(df2['ratioA'])*3
    # run Savgol Filter
    smooth_first = savgol_filter(smooth_first, 35, 3)
    smooth_last = savgol_filter(smooth_last, 35, 3)
    # retrim to original dataset
    smooth_first = smooth_first[len(df1['ratioA']):len(df1['ratioA'])*2]
    smooth_last = smooth_last[len(df2['ratioA']):len(df2['ratioA'])*2]

    # calculate the points where the ratio crosses 0.5
    first_crosses_up = []
    first_crosses_dw = []
    last_crosses_up = []
    last_crosses_dw = []
    # first
    for i in range(2, len(index)):
        if (smooth_first[i] > 0.5 and smooth_first[i-1] < 0.5):
            first_crosses_up.append(i)
        if (smooth_first[i] < 0.5 and smooth_first[i-1] > 0.5):
            first_crosses_dw.append(i)
    # last
    for i in range(2, len(index)):
        if (smooth_last[i] > 0.5 and smooth_last[i-1] < 0.5):
            last_crosses_up.append(i)
        if (smooth_last[i] < 0.5 and smooth_last[i-1] > 0.5):
            last_crosses_dw.append(i)
    
    # merge sets
    first_crosses = first_crosses_up + first_crosses_dw
    first_crosses.sort()
    last_crosses = last_crosses_up + last_crosses_dw
    last_crosses.sort()
    # correct for start and end
    if first_crosses_dw[0] < first_crosses_up[0]:
        first_crosses = [0] + first_crosses
    if first_crosses_up[-1] > first_crosses_dw[-1]:
        first_crosses = first_crosses + [366]
    if last_crosses_dw[0] < last_crosses_up[0]:
        last_crosses = [0] + last_crosses
    if last_crosses_up[-1] > last_crosses_dw[-1]:
        last_crosses = last_crosses + [366]
    # convert into tuples
    first_crosses_tuple = zip(first_crosses[0::2], first_crosses[1::2])
    last_crosses_tuple = zip(last_crosses[0::2], last_crosses[1::2])

    # calculate difference in onset and offset of EAC
    # "+ 1" to values becuase index is zero indexed and not the actual day of the year
    # first
    mask = smooth_first < 0.5
    boolcount = np.bincount((~mask).cumsum()[mask])
    first_offon = max((list(y) for (x,y) in itertools.groupby((enumerate(mask)),operator.itemgetter(1)) if x), key=len)
    first_start = first_offon[-1][0] + 1
    first_end = first_offon[0][0] + 1
    # correct start if beyond 365
    if first_start == 366:
        mask = smooth_first > 0.5
        boolcount = np.bincount((~mask).cumsum()[mask])
        first_offon = max((list(y) for (x,y) in itertools.groupby((enumerate(mask)),operator.itemgetter(1)) if x), key=len)
        first_start = first_offon[0][0] + 1 + 366
    # last
    mask = smooth_last < 0.5
    boolcount = np.bincount((~mask).cumsum()[mask])
    last_offon = max((list(y) for (x,y) in itertools.groupby((enumerate(mask)),operator.itemgetter(1)) if x), key=len)
    last_start = last_offon[-1][0] + 1
    last_end = last_offon[0][0] + 1
    # correct start if beyond 366 (last day of year)
    if last_start == 366:
        mask = smooth_last > 0.5
        boolcount = np.bincount((~mask).cumsum()[mask])
        last_offon = max((list(y) for (x,y) in itertools.groupby((enumerate(mask)),operator.itemgetter(1)) if x), key=len)
        last_start = last_offon[0][0] + 1 + 366

    # colors
    blue = '#0052cc'
    red = '#e60000'

    # make plot
    fig, ax = plt.subplots(figsize=(20, 3))
    plt.grid(ls='dashed', alpha=0.7)
    ax.stackplot(index, list(df_mean['ratioA']), color='#808080')
    # make bands
    ub = df_mean['ratioA'] + df_std['ratioA']
    lb = df_mean['ratioA'] - df_std['ratioA']
    ub = [1 if x >= 1 else x for x in ub]
    lb = [0 if x <= 0 else x for x in lb]
    # shade between bands
    # plt.fill_between(index, lb, ub, alpha=0.25, color='#4682B4')
    # add first and last 5 year lines
    plt.plot(index, smooth_first, '--', color=blue, alpha=0.6)
    plt.plot(index, smooth_last, '--', color=red, alpha=0.6)
    # add corssovers
    for cross in last_crosses_tuple:
        ax.stackplot(index[cross[0]:cross[1]], smooth_last[cross[0]:cross[1]], color=red, alpha=0.2)
    for cross in first_crosses_tuple:
        ax.stackplot(index[cross[0]:cross[1]], smooth_first[cross[0]:cross[1]], color=blue, alpha=0.2)

    # title
    plt.ylabel('Classification Ratio', labelpad=16, size = 14)
    plt.title(title, size=15)
    # edit axis
    X = plt.gca().xaxis
    X.set_major_locator(mdates.MonthLocator())
    X.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_xlim(datetime(2000, 1, 1),datetime(2000, 12, 31))

    # report data
    print('\nAnalysis report for "'+title+'" (days)')
    print('First 5 years: '+str(len(df1.ratioA[df1.ratioA > 0.5])))
    print('Last 5 years: '+str(len(df2.ratioA[df2.ratioA > 0.5])))
    print('Difference: '+str(len(df2.ratioA[df2.ratioA > 0.5]) - len(df1.ratioA[df1.ratioA > 0.5])))
    print('Mean dominance period length:'+str(len(df_mean.ratioA[df_mean.ratioA > 0.5])))
    print('Current Start difference: '+str(last_start - first_start))
    print('Current End difference: '+str(last_end - first_end))

    # save plot
    print('\nMaking plot..')
    plt.show()
    fig.savefig(out_fn+'.png')
    plt.close("all")

###########################################################
# Analyse change in the monthly mean of current influence #
###########################################################
def temporal_analysis(df, title, out_fn):
    """
    Todo - modify text position so is not dependent on 1994
    """
    # calculate ratios
    df['ratioA'] = [A/(A+B) for A, B in zip(df['countA'], df['countB'])]
    df['ratioB'] = [B/(A+B) for A, B in zip(df['countA'], df['countB'])]
    # add date components
    df['day'] = [x.day for x in df['dt']]
    df['month'] = [x.month for x in df['dt']]
    df['year'] = [x.year for x in df['dt']]

    # montly plot data
    df_month = df.groupby(['year', 'month']).mean().reset_index()
    df_month['xtime'] = [datetime(int(x[1]['year']), int(x[1]['month']), 1, 0, 0, 0) for x in df_month.iterrows()]
    # yearly plot data
    df_year = df_month.groupby(['year']).mean().reset_index()
    df_year['xtime'] = [datetime(int(x[1]['year']), 1, 1, 0, 0, 0) for x in df_year.iterrows()]
    df_year['std'] = df_month.groupby(['year'])['ratioA'].std().reset_index(drop=True)
    # drop incomplete years
    year_months = df_month.year.value_counts()
    incomplete_years = year_months[year_months < 12].index[0]
    df_year = df_year[df_year.year != incomplete_years]

    # set x axis index
    x = list(range(0,len(df_month.xtime)))
    x = [i/12 + df_month.xtime.iloc[0].year for i in x]
    df_month['x'] = x
    df_year['x'] = df_year['year']

    # get regression paramaters
    slope, intercept, r_value, p_value, std_err = stats.linregress(range(0,len(df_year['ratioA'])),df_year['ratioA'])

    # colors
    blue = '#0052cc'

    # __Monthly_plot__
    fig, ax = plt.subplots(figsize=(20, 3))
    plt.grid(ls='dashed', alpha=0.7)
    ax.stackplot(list(df_month['x']), list(df_month['ratioA']), color='#bfbfbf')
    ax.set(xticks=list(range(df_month.xtime.iloc[0].year, df_month.xtime.iloc[-1].year+1, 2)))
    ax.set_xlim(df_month.xtime.iloc[0].year-1, df_month.xtime.iloc[-1].year+1)
    # add regression
    plt.plot(df_year['x'], df_year['ratioA'], color='#246cb9', alpha=0.6)
    ax = sns.regplot(x='x', y="ratioA", data=df_year, color='#3785d8', line_kws={'alpha':0.7}, scatter_kws={'alpha':0.4}, ci=95, truncate=True)
    # add regression formula
    plt.text(1994.4, 0.95, 'y = '+str('{:.3f}'.format(round(slope, 3)))+'*x + '+str('{:.2f}'.format(round(intercept, 2))))
    plt.text(1994.4, 0.85, 'R = '+str('{:.2f}'.format(round(r_value, 2))))
    # labels
    plt.ylabel('Classification Ratio', labelpad=16, size=14)
    plt.title(title, size=15)
    plt.show()
    fig.savefig(out_fn+'.png')
    plt.close("all")

    # report line_kws
    print('\nYeraly mean regression paramaters for '+str(title)+':')
    print('Slope:'+str(slope))
    print('Intercept:'+str(intercept))
    print('R:'+str(r_value))

#########################
# Merge images together #
#########################
def img_join(output_name, img_ls, direction='vertical'):
    # open images
    images = [PIL.Image.open(i) for i in img_ls]
    
    # Resize if needed
    if direction == 'vertical':
        h_ls = [i.size[0] for i in images]
        if len(set(h_ls)) != 1:
            print('Width dimensions are not equal.. resizing')
            min_h = min(h_ls)
            images = [i.resize((min_h, i.size[1])) for i in images] 
    if direction == 'horizontal':
        v_ls = [i.size[1] for i in images]
        if len(set(v_ls)) != 1:
            print('Height dimensions are not equal.. resizing')
            min_v = min(v_ls)
            images = [i.resize((i.size[0], min_v)) for i in images] 

    # stack
    if direction == 'vertical':
        imgs_comb = np.vstack(np.asarray(i) for i in images)
        imgs_comb = PIL.Image.fromarray(imgs_comb)
        imgs_comb.save(output_name)
    elif direction == 'horizontal':
        imgs_comb = np.hstack(np.asarray(i) for i in images)
        imgs_comb = PIL.Image.fromarray(imgs_comb)
        imgs_comb.save(output_name)    
    else:
        print('Error: invalid margin string')

    print('\nImages joined - saved to: '+output_name)

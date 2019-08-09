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

    # set title
    title = 'Analysis Regions'
    if len(box_ls) == 1:
        title = 'Analysis Region'

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

    # smooth the trend lines
    smooth_mean = savgol_filter(df_mean['ratioA'], 35, 3)
    smooth_first = savgol_filter(df1['ratioA'], 35, 3)
    smooth_last = savgol_filter(df2['ratioA'], 35, 3)

    # calculate the points where the ratio crosses 0.5
    first_crosses = []
    last_crosses = []
    # first
    for i in range(2, len(index)):
        if (smooth_first[i] > 0.5 and smooth_first[i-1] < 0.5):
            first_crosses.append(i)
        if (smooth_first[i] < 0.5 and smooth_first[i-1] > 0.5):
            first_crosses.append(i)
    # last
    for i in range(2, len(index)):
        if (smooth_last[i] > 0.5 and smooth_last[i-1] < 0.5):
            last_crosses.append(i)
        if (smooth_last[i] < 0.5 and smooth_last[i-1] > 0.5):
            last_crosses.append(i)
    # trim if multiple
    first_crosses = [first_crosses[0], first_crosses[-1]]

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
    ax.stackplot(index[0:last_crosses[0]], smooth_last[0:last_crosses[0]], color=red, alpha=0.2)
    ax.stackplot(index[last_crosses[1]:-1], smooth_last[last_crosses[1]:-1], color=red, alpha=0.2)
    ax.stackplot(index[0:first_crosses[0]], smooth_first[0:first_crosses[0]], color=blue, alpha=0.2)
    ax.stackplot(index[first_crosses[1]:-1], smooth_first[first_crosses[1]:-1], color=blue, alpha=0.2)
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
    print('Onset difference: '+str(first_crosses[1] - last_crosses[1]))
    print('End difference: '+str(first_crosses[0] - last_crosses[0]))

    # save plot
    print('\nMaking plot..')
    plt.show()
    fig.savefig(out_fn+'.png')
    plt.close("all")

###########################################################
# Analyse change in the monthly mean of current influence #
###########################################################
def temporal_analysis(df, title, out_fn):
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
    # get regression paramaters
    slope, intercept, r_value, p_value, std_err = stats.linregress(range(0,len(df_year['ratioA'])),df_year['ratioA'])

    # __Monthly_plot__
    fig, ax = plt.subplots(figsize=(20, 3))
    plt.grid(ls='dashed', alpha=0.7)
    x = list(range(0,len(df_month.xtime)))
    x = [i/12 + df_month.xtime.iloc[0].year for i in x]
    df_month['x'] = x
    ax.stackplot(list(df_month['x']), list(df_month['ratioA']), color='#808080')
    ax.set(xticks=list(range(df_month.xtime.iloc[0].year, df_month.xtime.iloc[-1].year+1, 2)))
    ax.set_xlim(df_month.xtime.iloc[0].year, df_month.xtime.iloc[-1].year+1)
    plt.ylabel('Classification Ratio', labelpad=16, size=14)
    plt.show()
    fig.savefig(out_fn+'_month.png')
    plt.close("all")

    # __yearly_means_plot__
    fig, ax = plt.subplots(figsize=(20, 2))
    ax.set(xticks=list(range(df_year.xtime.iloc[0].year,df_month.xtime.iloc[-1].year+1,2)))
    ax.set_xlim(df_year.xtime.iloc[0].year, df_month.xtime.iloc[-1].year+1)
    ax = sns.regplot(x='year', y="ratioA", data=df_year, color='b', line_kws={'alpha':0.4}, scatter_kws={'alpha':0}, ci=95, truncate=True)
    ax.set_ylabel('Classification Ratio', labelpad=16, size=14)
    plt.grid(ls='dashed', alpha=0.7)
    plt.plot('year', 'ratioA', data=df_year, color='#606060', marker='+', alpha=1)
    plt.yticks(np.arange(0.2, 0.8, step=0.2))
    ax.set_ylim(0.1,0.9)
    plt.title(title, size=15)
    plt.show()
    fig.savefig(out_fn+'_year.png')
    plt.close("all")

    # report line_kws
    print('\nYeraly mean regression paramaters for '+str(title)+':')
    print('Slope:'+str(slope))
    print('Intercept:'+str(intercept))
    print('R:'+str(r_value))

    # merge plots
    img_join(out_fn+'.png', [out_fn+'_year.png', out_fn+'_month.png'])

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

# monthly averaged stack diagram
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from datetime import timedelta
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates # for month ticks

# for debugging
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

#############
# __SETUP__ #
#############

parser = argparse.ArgumentParser(description=__doc__)
# add a positional writable file argument
parser.add_argument('input_csv_file', type=argparse.FileType('r'))
# parser.add_argument('output_csv_file', type=argparse.FileType('w')) 
args = parser.parse_args()

############
# __DATA__ #
############

# Read in stack data
data = pd.read_csv(args.input_csv_file, parse_dates = ['xtime'], 
                    infer_datetime_format = True) #Read as DateTime obsject
data['xyear'] = [x.year for x in data['xtime']]
data['xday'] = [x.day for x in data['xtime']]
#data['DoY'] = [x.timetuple().tm_yday for x in data['xtime']]
data = data.drop_duplicates('xtime').reset_index(drop=True)
data = data.sort_values('xtime').reset_index(drop=True)

# group data
data2 = data.groupby(['xyear', 'xmonth']).mean().reset_index()
data2['xtime'] = [datetime(int(x[1]['xyear']), int(x[1]['xmonth']), 1, 0, 0, 0) for x in data2.iterrows()]

data3 = data2.groupby(['xyear']).mean().reset_index()
data3['xtime'] = [datetime(int(x[1]['xyear']), 1, 1, 0, 0, 0) for x in data3.iterrows()]

# Seasonal data
data4 = data[data.xyear != 2016]
data4 = data.groupby(['xmonth', 'xday']).mean().reset_index()
data4_std = data.groupby(['xmonth', 'xday']).std().reset_index()
# add seasonal first half vs second half
data4a = data[data.xyear != 2016]
data4a = data4a.set_index(['xtime'])
data4b = data4a
data4a = data4a.loc['1994-1-1':'1999-12-31']
data4b = data4b.loc['2010-1-1':'2015-12-31']
data4a = data4a.groupby(['xmonth', 'xday']).mean().reset_index()
data4b = data4b.groupby(['xmonth', 'xday']).mean().reset_index()

# % of year above 50% EAC dominance
data['dom'] = [1 if x > 0.5 else 0 for x in data['yEACr']]
data5 = data.groupby(['xyear', 'dom']).count()
dom1 = list(data5['yEAC'][1::2])
dom0 = list(data5['yEAC'][::2])
domtot = [x + y for x, y in zip(dom1, dom0)]
dom = [x/y for x, y in zip(dom1, domtot)]
dom[-1] = np.nan
data5 = {'dom':dom, 'year':list(range(1994,2017,1))}
data5 = pd.DataFrame(data=data5)

############
# __PLOT__ #
############

# # __normal_plot__
# fig, ax = plt.subplots(figsize=(20, 3))
# ax.stackplot(list(data['xtime']), list(data['yTSWr']), list(data['yEACr']))
# plt.title('EAC Influence (daily)')
# plt.show()
# plt.close("all")

# # __Monthly_plot__
# fig, ax = plt.subplots(figsize=(20, 3))
# ax.stackplot(list(data2['xtime']), list(data2['yTSWr']), list(data2['yEACr']))
# plt.title('EAC Influence (monthly)')
# plt.show()
# plt.close("all")

# # __yearly_means_plot__
# # v1
# fig, ax = plt.subplots()
# ax.stackplot(list(data3['xtime']), list(data3['yTSWr']), list(data3['yEACr']))
# plt.title('EAC Influence (yearly)')
# plt.show()
# plt.close("all")
# # v2
# p1 = plt.bar(list(data3['xyear']), list(data3['yTSWr']), 1)
# p2 = plt.bar(list(data3['xyear']), list(data3['yEACr']), 1, bottom=list(data3['yTSWr']))
# plt.ylabel('Relative Influence')
# plt.xlabel('Year')
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.xticks(np.arange(1994, 2017, 2))
# plt.title('EAC Influence (yearly)')
# plt.legend((p1[0], p2[0]), ('non-EAC', 'EAC'))
# plt.show()

# # __Seasonal_Plot__
# fig, ax = plt.subplots()
# ax.stackplot(list(data4.index), list(data4['yTSWr']), list(data4['yEACr']))
# plt.title('Mean Seasonal EAC Influence')
# plt.show()
# plt.close("all")

# # __Monthly_plot__
# fig, ax = plt.subplots(figsize=(20, 3))
# plt.grid(ls='dashed', alpha=0.7)
# x = list(range(0,len(data2.xtime)))
# x = [i/12 + 1994 for i in x]
# data2['x'] = x
# ax.stackplot(list(data2['x']), list(data2['yEACr']), color='#808080')
# ax.set(xticks=list(range(1994,2017,2)))
# ax.set_xlim(1994,2017)
# plt.ylabel('EAC Fraction', labelpad=16, size=14)
# plt.show()
# fig.savefig(out_fn + 'monthly_' + out_data + '.png')
# plt.close("all")


# # __Dominance_plot__
# # with sns.axes_style("darkgrid"):
# fig, ax = plt.subplots(figsize=(20, 2))
# plt.grid(ls='dashed', alpha=0.7)
# plt.plot(data5.year, data5.dom, color='#606060', alpha=1, marker='+')
# ax = sns.regplot(x='year', y="dom", data=data5, color='b', marker="+", 
# 					line_kws={'alpha':0.4}, scatter_kws={'alpha':0}, ci=95)
# plt.ylabel('EAC Dominance', labelpad=16, size=14)
# plt.xlabel('')
# plt.xticks(np.arange(1994, 2017, step=2))
# ax.set_xlim(1994,2017)
# plt.yticks(np.arange(0.2, 0.8, step=0.2))
# ax.set_ylim(0.1,0.7)
# plt.show()
# fig.savefig(out_fn + 'dom_' + out_data + '.png')
# plt.close("all")

# # __yearly_means_plot__
# fig, ax = plt.subplots(figsize=(20, 2))
# ax.set(xticks=list(range(1994,2017,2)))
# ax.set_xlim(1994, 2017)
# ax = sns.regplot(x='xyear', y="yEACr", data=data3, color='b', line_kws={'alpha':0.4}, scatter_kws={'alpha':0}, ci=95, truncate=True)
# ax.set_ylabel('EAC Fraction', labelpad=16, size=14)
# plt.grid(ls='dashed', alpha=0.7)
# plt.plot('xyear', 'yEACr', data=data3, color='#606060', marker='+', alpha=1)
# plt.yticks(np.arange(0.2, 0.8, step=0.2))
# ax.set_ylim(0.1,0.9)
# plt.show()
# fig.savefig(out_fn + 'year_' + out_data + '.png')
# plt.close("all")


# # make time ticks
# index = data4.index
# base = datetime(2000, 1, 1, 0, 0, 0)
# index = [base + timedelta(int(x)) for x in index]
# # Set the locator
# locator = mdates.MonthLocator()  # every month
# # Specify the format - %b gives us Jan, Feb...
# fmt = mdates.DateFormatter('%b')

# # __Seasonal_plot__
# if out_data == 'off':
# 	title = 'Offshore Zone'
# if out_data == 'on':
# 	title = 'Coastal Zone'
# fig, ax = plt.subplots(figsize=(20, 3))
# plt.grid(ls='dashed', alpha=0.7)
# ax.stackplot(index, list(data4['yEACr']), color='#606060')
# ub, lb = data4['yEACr']+data4_std['yEACr'], data4['yEACr']-data4_std['yEACr'] 
# ub = [1 if x >= 1 else x for x in ub]
# lb = [0 if x <= 0 else x for x in lb]
# plt.fill_between(index, lb, ub, alpha=0.25, color='#4682B4')
# # plt.plot(index, ub, '--', color='k', alpha=0.1)
# # plt.plot(index, lb, '--', color='k', alpha=0.1)
# plt.plot(index, data4a.yEACr, '--', color='b', alpha=0.6)
# plt.plot(index, data4b.yEACr, '--', color='r', alpha=0.6)
# plt.ylabel('EAC Fraction', labelpad=16, size = 14)
# plt.title(title, size=15)
# X = plt.gca().xaxis
# X.set_major_locator(locator)
# X.set_major_formatter(fmt)
# ax.set_xlim(datetime(2000, 1, 1),datetime(2000, 12, 31))
# # ax.spines['right'].set_visible(False)
# # ax.spines['top'].set_visible(False)
# plt.show()
# fig.savefig(out_fn + 'season_' + out_data + '.png')
# plt.close("all")

# print stuff I want
year_stuff = data3['yEACr']
print('###### YEARLY FRACTION ###########')
print('Mean and SD')
print(year_stuff.mean())
print(year_stuff.std())
print('first mean')
print(year_stuff[0:4].mean())
print(year_stuff[0:4].std())
print('last mean')
print(year_stuff[len(year_stuff)-5:-1].mean())
print(year_stuff[len(year_stuff)-5:-1].std())

dom_stuff = data5['dom']
print('###### Dominance stuff ###########')
print('Mean and SD')
print(dom_stuff[0:-2].mean())
print(dom_stuff[0:-2].std())
print('first mean')
print(dom_stuff[0:4].mean())
print(dom_stuff[0:4].std())
print('last mean')
print(dom_stuff[len(dom_stuff)-6:-2].mean())
print(dom_stuff[len(dom_stuff)-6:-2].std())




"""
lachlanphillips (master) temporal_plots $ python trend_explorer.py /Users/lachlanphillips/Dropbox/PhD/Analysis/ocean_classification/data_stacks/onshore-stack_000-277_20180425-1814.csv 
ON
###### YEARLY FRACTION ###########
Mean and SD
0.4708536334656757
0.08940392435036257
first mean
0.3467319479070744
0.018786951719614507
last mean
0.5234988442330587
0.032982613692586094

###### Dominance stuff ###########
Mean and SD
0.4606436181778648
0.08499046277265909
first mean
0.35866831349651923
0.03748293363981839
last mean
0.5202148364398533
0.027481583714660696

lachlanphillips (master) temporal_plots $ python trend_explorer.py /Users/lachlanphillips/Dropbox/PhD/Analysis/ocean_classification/data_stacks/offshore-stack_000-277_20180425-1814.csv
OFF
###### YEARLY FRACTION ###########
Mean and SD
0.6280077265959352
0.1106578898332975
first mean
0.48709580424367277
0.06391164303402014
last mean
0.7036252939232932
0.06771141222250544

###### Dominance stuff ###########
Mean and SD
0.6143780365653262
0.11436603505034662
first mean
0.49286436110487314
0.06320607588791324
last mean
0.6892768919829328
0.0704644597127811

for std addition subtraction +/- squared then take final square root

mean dif frac: +/- 0.14

%Onshore
%Frac + 0.18 +/- 0.04
%Dom + 0.16 +/- 0.05 
%time + 1.9 +/- 0.6 months

%Off shore
%Frac + 0.22 +/- 0.09
%Dom + 0.20 +/- 0.09
%time + 2.4 months +/- 1 momnth
"""

import math

def add_std(a, b):
	x = math.sqrt(a**2 + b**2)
	return x

def sub_std(a, b):
	x = math.sqrt(a**2 - b**2)
	return x















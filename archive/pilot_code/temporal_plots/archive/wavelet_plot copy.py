# monthly averaged stack diagram
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import argparse
import datetime
from datetime import datetime
import numpy as np
import seaborn as sns
from wavelets import WaveletAnalysis
# from scipy import signal
import cmocean # oceanogrpahy colorscales - https://matplotlib.org/cmocean/

# for debugging
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

# range for floats
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

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
data3_std = data.groupby(['xyear']).std().reset_index()
data3['xtime'] = [datetime(int(x[1]['xyear']), 1, 1, 0, 0, 0) for x in data3.iterrows()]

data4 = data[data.xyear != 2016]
data4 = data.groupby(['xmonth', 'xday']).mean().reset_index()
data4_std = data.groupby(['xmonth', 'xday']).std().reset_index()

# % of year above 50% EAC dominance
data['dom'] = [1 if x > 0.5 else 0 for x in data['yEACr']]
data5 = data.groupby(['xyear', 'dom']).count()
dom1 = list(data5['yEAC'][1::2])
dom0 = list(data5['yEAC'][::2])
domtot = [x + y for x, y in zip(dom1, dom0)]
dom = [x/y for x, y in zip(dom1, domtot)]
dom[-1] = np.nan
dom = dom [0:-1]
data5 = {'dom':dom, 'year':list(range(1994,2016,1))}
data5 = pd.DataFrame(data=data5)

############
# __WAVE__ #
############

def normalize(data, a, b):
    return [(b-a)*(((x-data.min())/(data.max()-data.min())))+a for x in data]

# given a signal x(t)
dat = data5 # yearly
# dat = data2 # monthly
x = np.asarray(dat['dom'])
# and a sample spacing
dt = 1
wa = WaveletAnalysis(x, time=dat['year'], dt=dt)
# wavelet power spectrum
power = wa.wavelet_power
# power2 = power*1000 # old method
# power2 = power2-5
power2 = normalize(power, -5, 5)

# scales 
scales = wa.scales
# associated time vector
t = wa.time
# reconstruction of the original data
rx = wa.reconstruction()

############
# __PLOT__ #
############

# fig, ax = plt.subplots()
# t2 = t + 1994
# T, S = np.meshgrid(t, scales)
# plt.contourf(T, S, power, list(frange(0.000, 0.011, 0.0005)), 
# 	cmap=cmocean.cm.thermal)
# cbar = plt.colorbar()
# plt.contour(T, S, power, [0.006, 0.008], colors='black')
# ax.set_yscale('log')
# ax.set_yticks([2, 5, 10, 20])
# ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax.set_ylabel('Period (years)')
# plt.xticks(np.arange(1994, 2017, step=2))
# plt.gca().invert_yaxis()
# # fig.savefig('test_wavelet_power_spectrum.png')

# # shade the region between the edge and coi
# C, S = wa.coi
# ax.fill_between(x=C, y1=S, y2=scales.max(), color='gray', alpha=0.3)
# ax.set_xlim(t.min(), t.max())

# # plt.show()

fig1, ax1 = plt.subplots(figsize=(12,3))
T, S = np.meshgrid(t, scales)
plt.contourf(T, S, power2, list(frange(-5, 6, 1)), 
	cmap=cmocean.cm.thermal)
cbar = plt.colorbar(ticks=[-5, 0, 5])
cbar.set_label('Wavelet Power Spectrum', rotation=270, labelpad=16, size=13)
#plt.contour(T, S, power2, [2, 3], colors='black')
ax1.set_yscale('log')
ax1.set_ylim(2,8)
ax1.set_xlim(1994,2016)
ax1.set_yticks([2, 4, 8])
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.set_ylabel('Log Transformed Period (years)', labelpad=16, size=13)
plt.xticks(np.arange(1994, 2017, step=2))
plt.minorticks_off()
plt.gca().invert_yaxis()

# line CI
C, S2 = wa.coi
plt.plot(C, S2, '--', color='grey', alpha=1)


# # PLOT other plot
# # __Monthly_plot__
# fig2, ax2 = plt.subplots(figsize=(20, 3))
# plt.grid(ls='dashed', alpha=0.7)
# x = list(range(0,len(data2.xtime)))
# x = [i/12 + 1994 for i in x]
# data2['x'] = x
# ax2.stackplot(list(data2['x']), list(data2['yEACr']), color='#808080')
# plt.title('EAC Influence (monthly)')
# ax2.set(xticks=list(range(1994,2017,2)))

# # __yearly_means_plot__
# fig2, ax2 = plt.subplots(figsize=(9.65, 2))
# ax2.set(xticks=list(range(1994,2017,2)))
# ax2.set_xlim(1994, 2016)
# ax2 = sns.regplot(x='xyear', y="yEACr", data=data3, color='k', scatter_kws={'alpha':0}, ci=95)
# ax2.set_ylabel('EAC Fraction', labelpad=16)
# plt.grid(ls='dashed', alpha=0.7)
# plt.plot('xyear', 'yEACr', data=data3, color='grey', marker='+', alpha=0.7, mew=2)
# plt.show()

# plt.show()
# plt.close("all")

# __Dominance_plot__
# with sns.axes_style("darkgrid"):
fig2, ax2 = plt.subplots(figsize=(9.65, 2))
plt.grid(ls='dashed', alpha=0.7)
plt.plot(data5.year, data5.dom, color='#606060', alpha=1, marker='+')
ax2 = sns.regplot(x='year', y="dom", data=data5, color='b', marker="+", 
					line_kws={'alpha':0.4}, scatter_kws={'alpha':0}, ci=95)
plt.ylabel('EAC Dominance', labelpad=31, size=13)
plt.xlabel('')
plt.xticks(np.arange(1994, 2017, step=2))
ax2.set_xlim(1994,2016)
plt.yticks(np.arange(0.2, 0.8, step=0.2))
ax2.set_ylim(0.1,0.7)

plt.show()
plt.close("all")


###########
#   FFT   #
###########





















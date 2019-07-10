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

# Read in temp/salt data
data = pd.read_csv(args.input_csv_file, parse_dates = ['xtime'], 
                    infer_datetime_format = True) #Read as DateTime obsject
data['xyear'] = [x.year for x in data['xtime']]
data['xmonth'] = [x.month for x in data['xtime']]

# monthly data
data2 = data.groupby(['xyear', 'xmonth']).mean().reset_index()
data2['xtime'] = [datetime(int(x[1]['xyear']), int(x[1]['xmonth']), 1, 0, 0, 0) for x in data2.iterrows()]

# yearly data
data2 = data.groupby(['xyear', 'xmonth']).mean().reset_index()
data2['xtime'] = [datetime(int(x[1]['xyear']), int(x[1]['xmonth']), 1, 0, 0, 0) for x in data2.iterrows()]

data3 = data2.groupby(['xyear']).mean().reset_index()
data3['xtime'] = [datetime(int(x[1]['xyear']), 1, 1, 0, 0, 0) for x in data3.iterrows()]


############
# __WAVE__ #
############

def normalize(data, a, b):
    return [(b-a)*(((x-data.min())/(data.max()-data.min())))+a for x in data]

# given a signal x(t)
dat = data3 # 3 = yearly, 2 = monthly
x = np.asarray(dat['full_temp'])
# and a sample spacing
dt = 1
wa = WaveletAnalysis(x, time=dat['xyear'], dt=dt)
# wavelet power spectrum
power = wa.wavelet_power
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

plt.show()






















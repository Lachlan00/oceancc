# monthly averaged stack diagram
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import datetime
from datetime import datetime
import numpy as np

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
data = data.drop_duplicates('xtime').reset_index(drop=True)
data = data.sort_values('xtime').reset_index(drop=True)

# group data
data2 = data.groupby(['xyear', 'xmonth']).mean().reset_index()
data2['xtime'] = [datetime(int(x[1]['xyear']), int(x[1]['xmonth']), 1, 0, 0, 0) for x in data2.iterrows()]

data3 = data2.groupby(['xyear']).mean().reset_index()
data3['xtime'] = [datetime(int(x[1]['xyear']), 1, 1, 0, 0, 0) for x in data3.iterrows()]

data4 = data[data.xyear != 2016]
data4 = data.groupby(['xmonth', 'xday']).mean().reset_index()

############
# __PLOT__ #
############

# __normal_plot__
fig, ax = plt.subplots(figsize=(20, 3))
ax.stackplot(list(data['xtime']), list(data['yTSWr']), list(data['yEACr']))
plt.title('EAC Influence (daily)')
plt.show()
plt.close("all")


# __Mmnthly_plot__
fig, ax = plt.subplots(figsize=(20, 3))
ax.stackplot(list(data2['xtime']), list(data2['yTSWr']), list(data2['yEACr']))
plt.title('EAC Influence (monthly)')
plt.show()
plt.close("all")

# __yearly_means_plot__
# v1
fig, ax = plt.subplots()
ax.stackplot(list(data3['xtime']), list(data3['yTSWr']), list(data3['yEACr']))
plt.title('EAC Influence (yearly)')
plt.show()
plt.close("all")
# v2
p1 = plt.bar(list(data3['xyear']), list(data3['yTSWr']), 1)
p2 = plt.bar(list(data3['xyear']), list(data3['yEACr']), 1, bottom=list(data3['yTSWr']))
plt.ylabel('Relative Influence')
plt.xlabel('Year')
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xticks(np.arange(1994, 2017, 2))
plt.title('EAC Influence (yearly)')
plt.legend((p1[0], p2[0]), ('non-EAC', 'EAC'))
plt.show()

# __Seasonal_Plot__
fig, ax = plt.subplots()
ax.stackplot(list(data4.index), list(data4['yTSWr']), list(data4['yEACr']))
plt.title('Mean Seasonal EAC Influence')
plt.show()
plt.close("all")



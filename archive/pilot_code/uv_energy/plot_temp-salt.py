# temp/salt plots

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import datetime
from datetime import datetime
import numpy as np
import pylab
import seaborn as sns

# for debugging
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

# plot input
plot_type = 'final'
# plot_type = str(input('What kind of plots? (scat/line/final)?: '))
# if plot_type != 'final':
# 	plot_input = str(input('What would you like to plot? (day/month/year)?: '))


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
# __PLOT__ #
############
# __LINE__ #
############

if plot_type == 'line':
	# DAILY
	if plot_input == 'day':
		# TEMP
		dat = data
		# __full_region__
		fig1 = plt.figure(figsize=(20,3))
		ax1 = fig1.add_subplot(111)
		ax1.plot(dat.xtime, dat.full_temp, color='red')
		plt.title('Full region temp')
		# __mont_region__
		fig2 = plt.figure(figsize=(20,3))
		ax2 = fig2.add_subplot(111)
		ax2.plot(dat.xtime, dat.mont_temp, color='red')
		plt.title('Mont region temp')
		# __bath_region__
		fig3 = plt.figure(figsize=(20,3))
		ax3 = fig3.add_subplot(111)
		ax3.plot(dat.xtime, dat.bath_temp, color='red')
		plt.title('Bath region temp')
		# SALT
		# __full_region__
		fig4 = plt.figure(figsize=(20,3))
		ax4 = fig4.add_subplot(111)
		ax4.plot(dat.xtime, dat.full_salt, color='blue')
		plt.title('Full region salt')
		# __mont_region__
		fig5 = plt.figure(figsize=(20,3))
		ax5 = fig5.add_subplot(111)
		ax5.plot(dat.xtime, dat.mont_salt, color='blue')
		plt.title('Mont region salt')
		# __bath_region__
		fig6 = plt.figure(figsize=(20,3))
		ax6 = fig6.add_subplot(111)
		ax6.plot(dat.xtime, dat.bath_salt, color='blue')
		plt.title('Bath region salt')

		plt.show()
		plt.close("all")

	# MONTHLY
	if plot_input == 'month':
		# TEMP
		dat = data2
		# __full_region__
		fig1 = plt.figure(figsize=(20,3))
		ax1 = fig1.add_subplot(111)
		ax1.plot(dat.xtime, dat.full_temp, color='red')
		plt.title('Full region temp')
		# __mont_region__
		fig2 = plt.figure(figsize=(20,3))
		ax2 = fig2.add_subplot(111)
		ax2.plot(dat.xtime, dat.mont_temp, color='red')
		plt.title('Mont region temp')
		# __bath_region__
		fig3 = plt.figure(figsize=(20,3))
		ax3 = fig3.add_subplot(111)
		ax3.plot(dat.xtime, dat.bath_temp, color='red')
		plt.title('Bath region temp')
		# SALT
		# __full_region__
		fig4 = plt.figure(figsize=(20,3))
		ax4 = fig4.add_subplot(111)
		ax4.plot(dat.xtime, dat.full_salt, color='blue')
		plt.title('Full region salt')
		# __mont_region__
		fig5 = plt.figure(figsize=(20,3))
		ax5 = fig5.add_subplot(111)
		ax5.plot(dat.xtime, dat.mont_salt, color='blue')
		plt.title('Mont region salt')
		# __bath_region__
		fig6 = plt.figure(figsize=(20,3))
		ax6 = fig6.add_subplot(111)
		ax6.plot(dat.xtime, dat.bath_salt, color='blue')
		plt.title('Bath region salt')

		plt.show()
		plt.close("all")

	# YEARLY
	if plot_input == 'year':
		# TEMP
		dat = data3
		# __full_region__
		fig1 = plt.figure(figsize=(20,3))
		ax1 = fig1.add_subplot(111)
		ax1.plot(dat.xtime, dat.full_temp, color='red')
		plt.title('Full region temp')
		# __mont_region__
		fig2 = plt.figure(figsize=(20,3))
		ax2 = fig2.add_subplot(111)
		ax2.plot(dat.xtime, dat.mont_temp, color='red')
		plt.title('Mont region temp')
		# __bath_region__
		fig3 = plt.figure(figsize=(20,3))
		ax3 = fig3.add_subplot(111)
		ax3.plot(dat.xtime, dat.bath_temp, color='red')
		plt.title('Bath region temp')
		# SALT
		# __full_region__
		fig4 = plt.figure(figsize=(20,3))
		ax4 = fig4.add_subplot(111)
		ax4.plot(dat.xtime, dat.full_salt, color='blue')
		plt.title('Full region salt')
		# __mont_region__
		fig5 = plt.figure(figsize=(20,3))
		ax5 = fig5.add_subplot(111)
		ax5.plot(dat.xtime, dat.mont_salt, color='blue')
		plt.title('Mont region salt')
		# __bath_region__
		fig6 = plt.figure(figsize=(20,3))
		ax6 = fig6.add_subplot(111)
		ax6.plot(dat.xtime, dat.bath_salt, color='blue')
		plt.title('Bath region salt')

		plt.show()
		plt.close("all")

############
# __PLOT__ #
############
# __SCAT__ #
############

if plot_type == 'scat':
	# DAILY
	if plot_input == 'day':
		# TEMP
		dat = data
		# __full_region__
		fig1 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax1 = sns.regplot(x='x', y="full_temp", data=dat, color='r')
		plt.title('Full region temp')
		# __mont_region__
		fig2 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax2 = sns.regplot(x='x', y="mont_temp", data=dat, color='r')
		plt.title('Mont region temp')
		# __bath_region__
		fig3 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax3 = sns.regplot(x='x', y="bath_temp", data=dat, color='r')
		plt.title('Bath region temp')
		# SALT
		# __full_region__
		fig4 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax4 = sns.regplot(x='x', y="full_salt", data=dat, color='b')
		plt.title('Full region salt')
		# __mont_region__
		fig5 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax5 = sns.regplot(x='x', y="mont_salt", data=dat, color='b')
		plt.title('Mont region salt')
		# __bath_region__
		fig6 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax6 = sns.regplot(x='x', y="bath_salt", data=dat, color='b')
		plt.title('Bath region salt')

		plt.show()
		plt.close("all")

	# MONTHLY
	if plot_input == 'month':
		# TEMP
		dat = data2
		# __full_region__
		fig1 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax1 = sns.regplot(x='x', y="full_temp", data=dat, color='r', marker="+")
		plt.title('Full region temp')
		# __mont_region__
		fig2 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax2 = sns.regplot(x='x', y="mont_temp", data=dat, color='r', marker="+")
		plt.title('Mont region temp')
		# __bath_region__
		fig3 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax3 = sns.regplot(x='x', y="bath_temp", data=dat, color='r', marker="+")
		plt.title('Bath region temp')
		# SALT
		# __full_region__
		fig4 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax4 = sns.regplot(x='x', y="full_salt", data=dat, color='b', marker="+")
		plt.title('Full region salt')
		# __mont_region__
		fig5 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax5 = sns.regplot(x='x', y="mont_salt", data=dat, color='b', marker="+")
		plt.title('Mont region salt')
		# __bath_region__
		fig6 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax6 = sns.regplot(x='x', y="bath_salt", data=dat, color='b', marker="+")
		plt.title('Bath region salt')

		plt.show()
		plt.close("all")

	# YEARLY
	if plot_input == 'year':
		# TEMP
		dat = data3
		# __full_region__
		fig1 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax1 = sns.regplot(x='x', y="full_temp", data=dat, color='r', marker="+")
		plt.title('Full region temp')
		# __mont_region__
		fig2 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax2 = sns.regplot(x='x', y="mont_temp", data=dat, color='r', marker="+")
		plt.title('Mont region temp')
		# __bath_region__
		fig3 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax3 = sns.regplot(x='x', y="bath_temp", data=dat, color='r', marker="+")
		plt.title('Bath region temp')
		# SALT
		# __full_region__
		fig4 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax4 = sns.regplot(x='x', y="full_salt", data=dat, color='b', marker="+")
		plt.title('Full region salt')
		# __mont_region__
		fig5 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax5 = sns.regplot(x='x', y="mont_salt", data=dat, color='b', marker="+")
		plt.title('Mont region salt')
		# __bath_region__
		fig6 = plt.figure(figsize=(20,3))
		dat['x'] = list(range(0,len(dat.xtime)))
		ax6 = sns.regplot(x='x', y="bath_salt", data=dat, color='b', marker="+")
		plt.title('Bath region salt')

		plt.show()
		plt.close("all")

if plot_type == 'final':
	# TEMP
	dat = data2
	with sns.axes_style("darkgrid"):
		# __mont_region__
		fig1 = plt.figure(figsize=(20,3))
		x = list(range(0,len(dat.xtime)))
		x = [i/12 + 1994 for i in x]
		dat['x'] = x
		plt.plot( 'x', 'bath_temp', data=dat, color='grey', marker='+', alpha=0.4, mew=2)
		ax1 = sns.regplot(x='x', y="bath_temp", data=dat, color='r', marker="+", scatter_kws={'alpha':0}, ci=95)
		plt.title('Monthly Mean Temperature', size=15)
		ax1.set(xticks=list(range(1994,2017,2)))
		ax1.set(xlabel='')
		plt.ylabel('Temperature ($^\circ$C)', labelpad=16, size = 14)
		# __mont_region__
		fig2 = plt.figure(figsize=(20,3))
		x = list(range(0,len(dat.xtime)))
		x = [i/12 + 1994 for i in x]
		dat['x'] = x
		plt.plot( 'x', 'bath_salt', data=dat, color='grey', marker='+', alpha=0.4, mew=2)
		ax2 = sns.regplot(x='x', y="bath_salt", data=dat, color='g', marker="+", scatter_kws={'alpha':0}, ci=95)
		plt.title('Monthly Mean Salinity', size=15)
		ax2.set(xticks=list(range(1994,2017,2)))
		ax2.set(xlabel='')
		plt.ylabel('Salinity (PSU)', labelpad=16, size = 14)

		plt.show()



# PRODUCE MOMNTAGUE STACK GIF

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap # basemap tools
import argparse
import datetime
from datetime import datetime
import pandas as pd
import numpy as np

# animation modules
import moviepy.editor as mpy # creates animation
from moviepy.video.io.bindings import mplfig_to_npimage # converts map to numpy array
from matplotlib.backends.backend_agg import FigureCanvasAgg # draws canvas so that map can be converted

# __SETUP__

parser = argparse.ArgumentParser(description=__doc__)
# add a positional writable file argument
parser.add_argument('input_csv_file', type=argparse.FileType('r'))
# parser.add_argument('output_csv_file', type=argparse.FileType('w')) 
args = parser.parse_args()

# load data from argparse
data = pd.read_csv(args.input_csv_file, parse_dates = ['xtime'], 
                    infer_datetime_format = True) #Read as DateTime obsject
data['xyear'] = [x.year for x in data['xtime']]
data['xday'] = [x.day for x in data['xtime']]# group data
data2 = data.groupby(['xyear', 'xmonth']).mean().reset_index()
data2['xtime'] = [datetime(int(x[1]['xyear']), int(x[1]['xmonth']), 1, 0, 0, 0) for x in data2.iterrows()]

sort_data = data.sort_values('xtime').reset_index(drop=True)
sort_data

# plot function
def plot_tracker(idx):
	plt.close("all")
	fig, ax = plt.subplots(figsize=(20, 3))
	ax.stackplot(list(data2['xtime']), list(data2['yTSWr']), list(data2['yEACr']))
	plt.title('EAC Influence (monthly) | frame: ' + str(idx).zfill(4))
	plt.axvline(x=datetime(sort_data['xtime'][idx].year, sort_data['xtime'][idx].month, 1), 
		color='r', linewidth=4)
	#convert to array
	canvas = FigureCanvasAgg(fig)
	canvas.draw()
	frame = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
	frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

	return frame

# __Animate__
out_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/gifs/full_run/EAC_animations/"
out = out_directory+'stack_track.gif'

frame_count = len(sort_data)
animation = mpy.VideoClip(plot_tracker, duration=frame_count)
animation.write_gif(out, fps=1)
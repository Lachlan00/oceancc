# Get data for analysis

from train import *
from classify import *
from data_processes import *
from data_visulisation import *

#################
# Configuration #
#################
# directories
CARS_directory = '/Users/lachlanphillips/PhD_Large_Data/CARS'
ROMS_directory = '/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset'
# Training sources
sourceboxA = [155, 160, -28.5, -22.5]
sourceboxB = [155, 160, -46, -41]
# Study zones (center point and radius)
JervBox = boxmaker(150.9451487, -35.113566, 50)
BateBox = boxmaker(150.2269, -36.25201, 50)
HoweBox = boxmaker(150.2925979, -37.586966, 50)

# check study zone box positions
check_boxROMS([JervBox, BateBox, HoweBox], ROMS_directory, 
			depthmax=4000, save=True, out_fn='./plots/study_zones.png')

# produce training data
train_CARS(CARS_directory, './data/', sourceboxA, sourceboxB, plot_boxes=True)

# build model
lr_model = current_model('./data/training_data.csv')

# get count data for regions
count_Jarv = analyse_region_counts(ROMS_directory, lr_model, JervBox, depthmax=4000)
count_Bate = analyse_region_counts(ROMS_directory, lr_model, BateBox, depthmax=4000)
count_Howe = analyse_region_counts(ROMS_directory, lr_model, HoweBox, depthmax=4000)

# save data
count_Jarv.to_csv('./data/count_Jerv.csv', index=False)
count_Bate.to_csv('./data/count_Bate.csv', index=False)
count_Howe.to_csv('./data/count_Howe.csv', index=False)
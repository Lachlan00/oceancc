# Get data for analysis

from train import *
from classify import *
from data_processes import *
from data_visulisation import *

###################
# Study zone plot #
###################
# directories
ROMS_directory = '/Volumes/LP_MstrData/master-data/ocean/ROMS/Montague_subset/'
# Study zones (center point and radius)
JervBox = boxmaker(150.9451487, -35.113566, 50)
BateBox = boxmaker(150.2269, -36.25201, 50)
HoweBox = boxmaker(150.2925979, -37.586966, 50)

# check study zone box positions
# check_boxROMS([JervBox, BateBox, HoweBox], ROMS_directory, 
#             depthmax=4000, save=True, out_fn='./plots/study_zones.png',
#             labels=['Jervis Bay', 'Batemans Bay', 'Cape Howe'])

# make map insert
#eac_panel(ROMS_directory, out_fn='./plots/eac_panel.png')
#map_inset(ROMS_directory, out_fn='./plots/map_inset_study.png')
#map_inset(ROMS_directory, out_fn='./plots/map_inset_eac.png', eac_panel_inset=True)
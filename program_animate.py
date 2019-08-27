from animate import *

# Config
ROMS_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset/"
output_directory = "/Users/lachlanphillips/Development/PhD/repos/oceancc-animations/"

# make temp
animateROMS(ROMS_directory, output_directory, 'temp')
# make salt
animateROMS(ROMS_directory, output_directory, 'salt')
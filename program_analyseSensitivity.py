from data_processes import *
from data_visulisation import *
from sensitivity import *
import json

#############
#__Config__ #
#############
# Analysis name
name = 'TAS-Box-N'

#############
# read meta #
#############
# read in JSON meta
meta = json.loads(open('./data/sensitivity/'+name+'/meta.json').read())
# set params
increment = meta["increment"]
iterations = meta["iterations"]
modA_NS, modA_EW = meta["modA_NS"], meta["modA_EW"]
modB_NS, modB_EW = meta["modB_NS"], meta["modB_EW"]

# Training sources
CARS_directory = '/Users/lachlanphillips/PhD_Large_Data/CARS/'
sourceboxA = meta["sourceboxA"]
sourceboxB = meta["sourceboxB"]
region = meta["region"]
#__Modification
modA_NS = meta["modA_NS"]
modA_EW = meta["modA_EW"]
modB_NS = meta["modB_NS"]
modB_EW = meta["modB_EW"]

################
# modification #
################
sourcebox_modA = boxMod(sourceboxA, increment, iterations, modA_NS, modA_EW)
sourcebox_modB = boxMod(sourceboxB, increment, iterations, modB_NS, modB_EW)

############
# make map #
############
sensitivity_analysisMap(name, CARS_directory, sourcebox_modA, sourcebox_modB, region, increment=increment, iterations=iterations, save=True)

#################
# make joy plot #
#################
sensitivity_analysisLine(name, './data/sensitivity/', iterations=iterations, save=True)

# join plots together (temp hack)
img_ls = ['./data/sensitivity/'+name+'/plots/sensitivity_joy.png', './data/sensitivity/'+name+'/plots/sensitivity_map.png']
img_join('./data/sensitivity/'+name+'/plots/sensitivity_results.png', img_ls, direction='horizontal')
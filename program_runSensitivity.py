from train import *
from classify import *
from data_processes import *
from data_visulisation import *
from sensitivity import *
import json

#############
#__Config__ #
#############
# Analysis name
name = 'EAC-Box-E'
# directories
CARS_directory = '/Users/lachlanphillips/PhD_Large_Data/CARS/'
ROMS_directory = '/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset/'
# Training sources
sourceboxA = [153.5, 155.5, -27.5, -24]
sourceboxB = [155, 160, -46, -41]
# set study zone (bate)
region = boxmaker(150.2269, -36.25201, 50)

#__Modification
modA_NS = 'NA'
modA_EW = 'E'
modB_NS = 'NA'
modB_EW = 'NA'
increment = 0.5
iterations = 5

###################
# do modification #
###################
sourcebox_modA = boxMod(sourceboxA, increment, iterations, modA_NS, modA_EW)
sourcebox_modB = boxMod(sourceboxB, increment, iterations, modB_NS, modB_EW)

################################
# Get sensitvity training data #
################################
sensitivity_obtainData(name, CARS_directory, './data/sensitivity/', sourcebox_modA, sourcebox_modB, increment=increment, iterations=iterations)

##############################
# Get sensitivity count data #
##############################
sensitivity_runModels(name, ROMS_directory, './data/sensitivity/', region)

# save meta
meta = {
	"name":name,
	"sourceboxA":sourceboxA,
	"sourceboxB":sourceboxB,
	"region":region,
	"increment":increment,
	"iterations":iterations,
	"modA_NS":modA_NS,
	"modA_EW":modA_EW,
	"modB_NS":modB_NS,
	"modB_EW":modB_EW,
}

with open('./data/sensitivity/'+name+'/meta.json', 'w') as fp:
    json.dump(meta, fp)
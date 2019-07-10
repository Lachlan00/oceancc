"""
Training data aggregator
Concats trainging data csv files
"""

import pandas as pd
from os import listdir
from os.path import isfile, join

print('Please wait: Aggregating training datasets...\n')

# read in data
directory = "../"
file_ls = [f for f in listdir(directory) if isfile(join(directory, f))]
file_ls = list(filter(lambda x:'train-data_seed' in x, file_ls)) # filter out any strange data in the folder

# load in and save each file to list
dat_ls = []
for fn in file_ls:
	data = pd.read_csv(directory + fn)
	dat_ls.append(data)

output = pd.concat(dat_ls)

output_fn = 'train-data_all.csv'
output.to_csv(output_fn, index=False)

print(str(len(file_ls)) + ' files aggregated into 1 dataset containing ' + str(len(output)) + ' rows.')
print('Data saved.')
print('\n** Program End **')
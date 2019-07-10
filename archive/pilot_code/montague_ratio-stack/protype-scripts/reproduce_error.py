# Reproduce predict_proba() error
import pandas as pd
# import numpy as np
import random 

# set randomness seed
random.seed(1010)

# make zip
list1 = random.sample(range(1, 10), 5)
list2 = random.sample(range(1, 10), 5)
zip1 = zip(list1, list2) 
set1 = set(zip1)

# print the zipped lists
print(list(zip1))
# print the zipped lists a second time
print(list(zip1))

print('')

# print the set
print(set1)
# print the set a second time
print(set1)
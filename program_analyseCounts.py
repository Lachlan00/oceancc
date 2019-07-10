# analyse counts
import pandas as pd

# local modules
from data_visulisation import *

# read in data
count_Jarv = pd.read_csv('./data/count_Jarv.csv', parse_dates=['dt'],
                        infer_datetime_format=True)
count_Bate = pd.read_csv('./data/count_Bate.csv', parse_dates=['dt'],
                        infer_datetime_format=True)
count_Gabo = pd.read_csv('./data/count_Gabo.csv', parse_dates=['dt'],
                        infer_datetime_format=True)

# make plots
seasonal_change_analysis(count_Jarv, 'Jarvis Bay', './plots/seasonal_Jarv.png')
seasonal_change_analysis(count_Bate, 'Batesmen Bay', './plots/seasonal_Bate.png')
seasonal_change_analysis(count_Gabo, 'Gabo Island', './plots/seasonal_Gabo.png')  

# join 3 plots together (temp hack)
img_ls = ['./plots/seasonal_Jarv.png', './plots/seasonal_Bate.png', './plots/seasonal_Gabo.png']
img_join('./plots/seasonal_All.png', img_ls, direction='vertical')
# analyse counts
import pandas as pd

# local modules
from data_visulisation import *

# read in data
count_Jerv = pd.read_csv('./data/count_Jerv.csv', parse_dates=['dt'],
                        infer_datetime_format=True)
count_Bate = pd.read_csv('./data/count_Bate.csv', parse_dates=['dt'],
                        infer_datetime_format=True)
count_Howe = pd.read_csv('./data/count_Howe.csv', parse_dates=['dt'],
                        infer_datetime_format=True)

# make seasonality plots
seasonal_change_analysis(count_Jerv, 'Jervis Bay', './plots/seasonal_Jerv.png')
seasonal_change_analysis(count_Bate, 'Batemans Bay', './plots/seasonal_Bate.png')
seasonal_change_analysis(count_Howe, 'Cape Howe', './plots/seasonal_Howe.png')  
# join 3 plots together (temp hack)
img_ls = ['./plots/seasonal_Jerv.png', './plots/seasonal_Bate.png', './plots/seasonal_Howe.png']
img_join('./plots/seasonal_All.png', img_ls, direction='vertical')

# make monthly chnage plots
seasonal_change_analysis(count_Jerv, 'Jervis Bay', './plots/monthly_Jerv.png')
seasonal_change_analysis(count_Bate, 'Batemans Bay', './plots/monthly_Bate.png')
seasonal_change_analysis(count_Howe, 'Cape Howe', './plots/monthly_Howe.png')
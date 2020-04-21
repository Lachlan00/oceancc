# Perform model
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
from netCDF4 import Dataset
from progressbar import ProgressBar
import os

# user defined modules
from data_processes import *

###################################
# Build logistic regression model #
###################################
def current_model(trainData_dir, verbose=True):
    """
    Build current logistic regression classifier
    """
    if verbose:
        print('\nBuilding ocean current classification model.')
        print('loading dataset..')
    train_data = pd.read_csv(trainData_dir, parse_dates = ['datetime'],
                            infer_datetime_format = True)
    # add "day of year" (DoY) to dataset 
    train_data['DoY'] = [int(x.day) for x in train_data['datetime']]

    # standardise data and get fits to be used for later scaling
    if verbose:
        print('Standardising data..')
    scaler_temp = preprocessing.StandardScaler().fit(train_data[['temp']])
    scaler_salt = preprocessing.StandardScaler().fit(train_data[['salt']])
    # scale training dataset
    train_data['temp'] = scaler_temp.transform(train_data[['temp']])
    train_data['salt'] = scaler_salt.transform(train_data[['salt']])

    # fit logistic regression to the training data and cross validate
    if verbose:
        print('Cross validating logistic regression model..')
    lr_model = LogisticRegressionCV(cv=10)
    lr_model = lr_model.fit(train_data[['temp','salt','DoY']], np.ravel(train_data[['class']]))

    # report model feature importance
    if verbose:
        print('Model coeffecients')
        print(lr_model.coef_)
        print('Temp * SD')
        print(np.std(np.asarray(train_data['temp']), 0)*lr_model.coef_[0][0])
        print('Salt * SD')
        print(np.std(np.asarray(train_data['salt']), 0)*lr_model.coef_[0][1])

    # return model and standardisation data as dictionary
    return({"lr_model": lr_model, "scaler_temp": scaler_temp, "scaler_salt": scaler_salt})


######################################################################
# Obtain current classification probability counts from netCDF frame #
######################################################################
def current_prob_count(lr_model, fh, frame_time, frame_idx, eta_rho, xi_rho):
    # get temperature and salinity data in region of interest
    temp = fh.variables['temp'][frame_idx,29,:,:][eta_rho, xi_rho]
    salt = fh.variables['salt'][frame_idx,29,:,:][eta_rho, xi_rho]
    # place in data frame (ravel to 1d array)
    df = pd.DataFrame(data={'temp': temp.ravel(), 'salt': salt.ravel()})
    # add DoY
    df['DoY'] = int(frame_time.day)
    # remove missing values
    df = df.dropna()
    # scale data (using scaled data from built model)
    df.loc[:,'temp'] = lr_model['scaler_temp'].transform(df[['temp']])
    df.loc[:,'salt'] = lr_model['scaler_salt'].transform(df[['salt']])
    # get probabilities using model
    probs = lr_model['lr_model'].predict_proba(df[['temp','salt','DoY']])
    # separate results
    probA, probB = zip(*probs)
    # make array of probA (probB is opposite)
    probA = np.asarray(probA)
    # count A and B classificatons
    countA = np.count_nonzero(probA >= 0.5)
    countB = np.count_nonzero(probA < 0.5)
    
    # return row for data frame
    return [frame_time, countA, countB]

###############################
# Compute regional count data #
###############################
def analyse_region_counts(ROMS_directory, lr_model, region, depthmax=1e10, name_check='NA'):
    """
    depthmax in meters (arbitarily large by default)
    """
    if os.path.exists('./data/'+name_check):
        if yes_or_no('"./data/'+name_check+'" already exists. Would you like to produce a new training dataset?'):
            print('"./data/'+name_check+'" will be overwirtten.')
        else:
            print('Using previously extracted dataset.')
            return
    print('\nExtracting current classification count data..')
    # get ROMS netCDF file list
    file_ls = [f for f in listdir(ROMS_directory) if isfile(join(ROMS_directory, f))]
    file_ls = list(filter(lambda x:'.nc' in x, file_ls))
    file_ls = sorted(file_ls)

    # get lats and lons
    nc_file = ROMS_directory + file_ls[0]
    fh = Dataset(nc_file, mode='r')
    lats = fh.variables['lat_rho'][:] 
    lons = fh.variables['lon_rho'][:]
    bath = fh.variables['h'][:]
    ocean_time = fh.variables['ocean_time'][:]
    array_dimensions = lons.shape

    # combine lat and lon to list of tuples
    point_tuple = zip(lats.ravel(), lons.ravel(), bath.ravel())
    # iterate over tuple points and keep every point that is in box
    point_list = []
    j = 0
    for i in point_tuple:
        if region[2] <= i[0] <= region[3] and region[0] <= i[1] <= region[1] and i[2] < depthmax:
            point_list.append(j)
        j = j + 1

    # make point list into tuple list of array coordinates
    eta_rho = []
    xi_rho = []
    for i in point_list:
        eta_rho.append(int(i/array_dimensions[1]))
        xi_rho.append(int(i%array_dimensions[1]))

    # set up progress bar
    pbar = ProgressBar(max_value=len(file_ls))
    
    # create data frame to hold count data
    df_count = pd.DataFrame(np.nan, index=range(0,(len(file_ls)+1)*len(ocean_time)), columns=['dt', 'countA', 'countB'])
    # extract count data from each netCDF file
    idx = 0
    print('Classifying ocean currents in '+str(len(file_ls))+' netCDF files..')
    pbar.update(0)
    for i in range(0, len(file_ls)):
        # import file
        nc_file = ROMS_directory + '/' + file_ls[i]
        fh = Dataset(nc_file, mode='r')
        # extract time
        ocean_time = fh.variables['ocean_time'][:]
        # get data
        for j in range(0, len(ocean_time)):
            # get dt from ocean_time
            frame_time = oceantime_2_dt(ocean_time[j])
            # get counts and sub into data frame
            df_count.iloc[idx] = current_prob_count(lr_model, fh, frame_time, j, eta_rho, xi_rho)
            idx += 1
            # update progress
            pbar.update(i)

        # close file
        fh.close()

    # drop NaNs and return dataset
    return(df_count.dropna())


# lists to collect ratio data
yEAC = []
xtime = []

# for getting time data
def grab_sst_time(time_idx):
    """
    gets datetime object for sst map projection
    """
    dtcon_days = time[time_idx]
    dtcon_start = datetime(1990,1,1) # This is the "days since" part
    dtcon_delta = timedelta(dtcon_days/24/60/60) # Create a time delta object from the number of days
    dtcon_offset = dtcon_start + dtcon_delta # Add the specified number of days to 1990
    frame_time = dtcon_offset
    return frame_time

def grab_prob(time_idx):
    # make frame__idx an integer to avoid slicing errors
    frame_idx = int(time_idx)
    # get 'frame_time'
    frame_time = grab_sst_time(frame_idx)

    # set month of year
    MoY_val = int(frame_time.month)

    # get list of temperature and salinity values from subset
    temp = []
    salt = []

    point_list = zip(eta_rho,xi_rho)

    # append values
    for i, j in point_list:
        temp.append(fh.variables['temp'][frame_idx,29,i,j])
        salt.append(fh.variables['salt'][frame_idx,29,i,j])

    data = {'var1': temp, 'var2': salt}
    data = pd.DataFrame(data=data)
    data['MoY'] = MoY_val
    # remove masked floats
    data = data[data.var1 >= 1]

    # calculate probabilities
    probs = lr_model.predict_proba(data[['var1','var2','MoY']])
    prob_TSW, prob_EAC = zip(*probs)
    # convert tuples to list
    prob_EAC = list(prob_EAC)
    # make 1D array
    prob_EAC = np.asarray(prob_EAC)

    # calulcate ratio metric
    count_EAC = np.count_nonzero(prob_EAC > 0.5)
    count_TSW = np.count_nonzero(prob_EAC < 0.5)
    # add to lists
    yEAC.append(count_EAC)
    xtime.append(frame_time)

    # get data for each file (may take a while)
for i in range(start, end):
    # import file
    nc_file = in_directory + '/' + file_ls[i]
    fh = Dataset(nc_file, mode='r')
    print('Extracting data from file: '+file_ls[i]+' | '+str(i+1)+' of '+ str(len(file_ls)))
    # fname = str(file_ls[i])[11:16]

    # extract time
    time = fh.variables['ocean_time'][:]

    # get data
    for i in range(0, len(time)):
        grab_prob(i)
    # close file
    fh.close()
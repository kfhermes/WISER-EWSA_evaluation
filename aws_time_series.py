import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re
import numpy as np
import seaborn as sns
import chardet
import xarray as xr
from pyproj import Transformer
import sys
from datetime import datetime
import cartopy.crs as ccrs



def replace_spaces_with_underscore(directory, keyword):
    # Get a list of CSV files containing the keyword
    csv_files = [file for file in glob.glob(os.path.join(directory, f'*{keyword}*.csv')) if os.path.isfile(file)]

    for csv_file in csv_files:
        # Replace spaces with underscores in the filename
        new_filename = re.sub(r'\s', '_', os.path.basename(csv_file))
        new_filepath = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(csv_file, new_filepath)
        print(f'Renamed: {csv_file} -> {new_filepath}')

        return new_filepath

def convert_to_24_hour_format(time_str):
    time_parts = time_str.split()
    hours, minutes = map(int, time_parts[0].split(':'))
    am_pm = time_parts[1]

    if am_pm.upper() == 'PM' and hours < 12:
        hours += 12
    elif am_pm.upper() == 'AM' and hours == 12:
        hours = 0

    return '{:02d}:{:02d}'.format(hours, minutes)

# Custom key function to extract date and time from file path
def extract_datetime(file_path):
    match = re.search(r'\d{8}T\d{6}Z', file_path)
    if match:
        return datetime.strptime(match.group(), '%Y%m%dT%H%M%SZ')
    
def generate_array(data):

    min_lon = 20
    max_lon = 35
    min_lat = -18
    max_lat = -8

    proj = ccrs.Geostationary(central_longitude=0.0,satellite_height=35785831)
    transformer_from_latlon = Transformer.from_crs("EPSG:4326", proj.to_proj4(), always_xy=True)
   
    min_x, max_y = transformer_from_latlon.transform(min_lon, max_lat)
    max_x, min_y = transformer_from_latlon.transform(max_lon, min_lat)


    zam = data.sel(nx = slice(min_x, max_x), ny= slice(max_y, min_y))

    x_coords = zam.nx.values
    y_coords = zam.ny.values

    zam_array = np.flipud(zam['crr_intensity'].values)

    return zam, zam_array, x_coords, y_coords
    
################################### paths ##############################

path_to_aws = str(sys.argv[1])
station = str(sys.argv[2])

#put yesterday's date here in 'YYYY-MM-DD ' format (leave a space at the end)
crr_date = '2024-01-30 '


####################################METADATA#################################

path_to_metadata = './active_aws_metadata.csv'
meta = pd.read_csv(path_to_metadata)
active = meta[meta['Active'] == 'Y']

location = active[active.District.str.contains(station)]

print(location)

lat = location['Station Location'].values.astype('float32')[0]
lon = location['Unnamed: 10'].values.astype('float32')[0]

print(lon,lat)


######################### loading in aws data ##########################

print(path_to_aws)
aws_file = replace_spaces_with_underscore(path_to_aws, station)

with open(aws_file, 'rb') as f:
    result = chardet.detect(f.read())

df = pd.read_csv(aws_file, encoding=result['encoding'],delimiter=';')

# Remove parentheses and their contents from column names
df.rename(columns=lambda x: x.split('(')[0].strip(), inplace=True)
df.replace('*', np.nan, inplace=True)
df.drop(df.index[-1],inplace=True)

# Apply the conversion function to the entire column
df['Time'] = df['Time'].apply(lambda x: convert_to_24_hour_format(x))

# Assuming 'TmStamp' is a datetime column in your DataFrame
df['Time'] = pd.to_datetime(crr_date + df['Time'])

# Set the 'TmStamp' column as the index
df.set_index('Time', inplace=True)

if 'Precipitation' in df.columns:
    columns_to_plot = ['Precipitation','Temperature','Relative Humidity', 'Wind Speed', 'Wind Direction']
elif 'Accumulated NRT' in df.columns:
    columns_to_plot = ['Accumulated NRT','Temperature','Relative Humidity', 'Wind Speed', 'Wind Direction']
else:
    columns_to_plot = ['Accumulated Total NRT','Temperature','Relative Humidity', 'Wind Speed', 'Wind Direction']
    df['Accumulated Total NRT'] = df['Accumulated Total NRT'] - df['Accumulated Total NRT'].iloc[0]


############################### loading in CRR data ###############################

datestring = crr_date.replace('-', '').replace(' ', '')

path_to_crr = '/gws/nopw/j04/swift/WISER-EWSA/Leeds_CRR/data/'+datestring+'/CRR/'
crr_files = glob.glob(path_to_crr + '/*120.nc')

#print(crr_files)

path_to_crr_temp = '/gws/nopw/j04/swift/WISER-EWSA/Leeds_CRR/temp/'
extra_crr_files = glob.glob(path_to_crr_temp + '*' + datestring + '*120.nc')

#print(extra_crr_files)

crr_file_paths = crr_files + extra_crr_files

# Sort the list of file paths by date and time
sorted_file_paths = sorted(crr_file_paths, key=extract_datetime)

# Keep track of unique time steps
unique_time_steps = set()

# Filter out duplicates and create a new list
filtered_file_paths = []
for file_path in sorted_file_paths:
    time_step = extract_datetime(file_path)
    if time_step not in unique_time_steps:
        unique_time_steps.add(time_step)
        filtered_file_paths.append(file_path)

#print(filtered_file_paths)

crr = []

for i in range(len(filtered_file_paths)):
    data = xr.open_dataset(filtered_file_paths[i])
    crr.append(generate_array(data)[1])

crr = np.array(crr)


#find station location in crr index 

proj = ccrs.Geostationary(central_longitude=0.0,satellite_height=35785831)

transformer_from_latlon = Transformer.from_crs("EPSG:4326", proj.to_proj4(), always_xy=True)
   
x, y = transformer_from_latlon.transform(lon, lat)

dum = generate_array(xr.open_dataset(filtered_file_paths[0]))[0]

xx = dum.sel(nx=x,ny=y,method='nearest')['nx'].values
yy = dum.sel(nx=x,ny=y,method='nearest')['ny'].values

xi = np.where(dum['nx']==xx)[0][0]
yi = np.where(dum['ny']==yy)[0][0]

#create time array for crr data 
crr_times = []
for i in range(crr.shape[0]):
    crr_times.append(pd.to_datetime(filtered_file_paths[i][-23:-7]))

################################## FIGURE CREATION ###################################
# Create a subplot grid
fig, axes = plt.subplots(nrows=len(columns_to_plot)+1, ncols=1, figsize=(10, 12), sharex=True)

crr_ts1 = crr[:,yi-1:yi+1+1,xi-1:xi+1+1].max(axis=(1,2))
crr_ts2 = crr[:,yi-4:yi+4+1,xi-4:xi+4+1].max(axis=(1,2))
crr_ts3 = crr[:,yi-7:yi+7+1,xi-7:xi+7+1].max(axis=(1,2))
crr_ts4 = crr[:,yi-10:yi+10+1,xi-10:xi+10+1].max(axis=(1,2))

axes[0].plot(crr_times,crr_ts1,label='CRR9',c='r')
axes[0].plot(crr_times,crr_ts2,label='CRR27',c='b')
axes[0].plot(crr_times,crr_ts3,label='CRR45',c='g')
axes[0].plot(crr_times,crr_ts4,label='CRR63',c='k')
axes[0].set_ylabel('CRR')
axes[0].legend()

# Plot each time series in a separate subplot
for i, column in enumerate(columns_to_plot):
    try:

        axes[i+1].plot(df.index, df[column].astype('float64'), label=column)
        axes[i+1].set_ylabel(column)
        axes[i+1].legend()
    except:
        print('column not avaiable: ' + columns_to_plot[i])
# Set x-axis label for the last subplot
axes[-1].set_xlabel('Time')

# Adjust layout to prevent clipping of x-axis labels
plt.tight_layout()
plt.suptitle(station + ' ' + crr_date)
# Show the plot
plt.savefig(station + '_' + datestring + '.png')

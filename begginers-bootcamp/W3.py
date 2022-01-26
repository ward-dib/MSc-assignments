"""
Created on Thu Nov 12 10:42:02 2020

@author: wrddib

Weekly challenge W3

"""
# Import libraries needed.
import requests
import h5py
import numpy as np

# Define URL.
url = 'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Tmax/date/UK.txt'

# Assign the text of the webpage to a dataframe.
metdata = (requests.get(url)).text

print(metdata)

# Creating the HDF5 file.

# Create new file.
hf = h5py.File('metdata.hdf5', 'w')

#Close the file.
hf.close()

# Re-open it in read mode.
hf = h5py.File('metdata.hdf5', 'r')

# Access the data.
print(hf['metdata'])
print(hf['metdata'].shape)
print(np.array(hf['metdata']))

# Close it again.
hf.close()

# Creating a class.

class met:
    """ Metdata Docstring """
    year = 2000
    rainfall = np.zeros(12)
    sun = np.zeros(12)
    tmax = np.zeros(12)
    tmin = np.zeros(12)
    # Create dictionary for the month indexes.
    month_dict = {'jan':0, 'feb':1, 'mar':2, 'apr':3, 'may':4, 'jun':5,
                  'jul':6, 'aug':7, 'sep':8, 'oct':9, 'nov':10, 'dec':10}

    # Set attributes for each parameter.
    def __init__(self, year = 2000, notes = 'None'):
        self.year = year
        self.notes = notes

    def average_sun(self):
        return np.mean(self.sun)
    
    def set_sun(self,month,sun_hours):
        self.sun[self.month_dict[month]] = sun_hours
        
    def return_met_for_month(self,month):
       return self.rainfall[self.month_dict[month]]
   
    def extreme_temp(self):
        return np.max(self.tmax) - np.min(self.min)

# Select one to test.
year2004 = met(year = 2004, notes = 'High Temperatures')

#print(year2005.notes)

year2004.set_sun('apr', 123)

print(year2004.sun)
print(year2004.average_sun())
print(year2004.return_met_for_month('apr'))

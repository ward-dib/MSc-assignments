import requests

import numpy as np

data = (requests.get('https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Tmean/date/UK.txt')).text

n_corona = 0 

for d in data.split():
    if ('covid' in d):
        n_corona+=1
        
print(f"There are {n_corona} mentions today.")

----

__author__= "Ward Dib" 
__date__= "2020-11-12"
__version__= "1"

"""
Week 2 practical challenge: Question 1.

Read   the   data   at
https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets
/Tmean/date/UK.txt 
representing gridded mean air temperature data for the UK since 1884. 

Calculate:
    1.  The month and year of the most extreme temperatures on the record
    
"""

import requests
import numpy as np

# Define orignal url to pull the weather information from.
url = 'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Tmean/date/UK.txt'

# Get the text of the webpage.
data = (requests.get(url)).text

# Split into lines.
lines = data.split('\n')

# Declare two empty lists for month and year to store data as it is read.
year = []
month = []

# Loop over the lines between the specified indexes.

for l in lines[6:-1]:
    # Split the data based on whitespace and loop over it.
    cols = l.split()
    # print(cols) can be used to print all the columns of data.
    year.append(int(cols[0]))
    month.append([float(i) for i in cols[1:13]])

# Cast year and month as numpy arrays.

year = np.array(year)
month = np.array(month)

'''
We can use print(month.shape, year.shape)
here to pull all the years' temperature data.

'''

# Cast minimum and maximum temperatures as numpy arrays.

min_temp = np.min(month)
max_temp = np.max(month)

# Use unravel_index to convert index coordinates into matrix coordinates.
year_min, month_min = np.unravel_index(month.argmin(), month.shape)
year_max, month_max = np.unravel_index(month.argmax(), month.shape)

# Define a dictionary to assign an index to every month name.
month_dict = {0:"Jan", 1:"Feb", 2:"Mar", 3:"Apr", 4:"May", 5:"Jun",
              6:"Jul", 7:"Aug", 8:"Sep", 9:"Oct", 10:"Nov", 11:"Dec"}

print(f"Minimum temperature was {month[year_min,month_min]} degC in\
 {month_dict[month_min]} {year[year_min]}")

print(f"Minimum temperature was {month[year_max,month_max]} degC in\
 {month_dict[month_max]} {year[year_max]}")

----------------------------------------------------

__author__= "Ward Dib" 
__date__= "2020-11-12"
__version__= "2"

'''
Week 2 practical challenge: Question 2.

Parse the HTML of the ‘Astronomy Picture of the Day’ (APOD) website
https://apod.nasa.gov/apod/astropix.html to automatically download the
most recent picture. Crop out the central 256 × 256 pixels and save as
a local JPEG.

'''

from skimage import io
import numpy as np
import requests

import requests

# Define orignal url to apod.
url = 'https://apod.nasa.gov/apod/astropix.html'

# Get the text of the webpage.
data = requests.get(url).text

# print(data) can be used to check here.

# Split into lines.
lines = data.split('\n')

# Loop over all the lines.
for l in lines:
    # Search for html tag in IMG SRC.
    if "IMG SRC" in l :
        
        ''' print = (l) can be used here to fetch all the lines.
        split returns a list and it splits according to the position
        in the brackets .
        print(l.split('"')) can be used to find what's betweet quotation
        marks.
        '''
        
        # When found, split by quotation marks to get path.
        img = l.split('"')[1]
        # Don't need to search any further.
        break
    
# Define our new url to the source the image.
new_url = url.replace('astropix.html', img)

# print(new_url) can be used to get the url of the image as a simple jpg.

# Make image request
image_data = requests.get(new_url)

# Open a file in write binary mode.

filehandle = open('/users/wrdxo/desktop/apod.jpg', 'wb')

# Write the content of the request to the file.
filehandle.write(image_data.content)

# Close the file.
filehandle.close()


# Read into a numpy array usiing skimage.
image = io.imread('/users/wrdxo/desktop/apod.jpg')

# print(image.shape) can be used to get size and dimensions of the picture.

# Find coordinates of the central pixel.
xc = int(image.shape[0]/2)
yc = int(image.shape[1]/2)

print(xc,yc)

# Define a cropping window.
win = 128

'''
This defines a slice in one of the dimensions from the central pixel to 
the edge of image. This silces 128 pixels in each direction starting from
the central pixel, ie 128 up, 128 down, 128 left, 128 right. This accounts
to 256 pixels on the x axis and 256 on the y axis, which gives the desired 
dimensions.
'''

# Slice out the array in the first two axes.
cropped = image[xc-win:xc+win,yc-win:yc+win,:]

# Print(cropped.shape) can be used to to confirm it is the desired dimensions.
# Save the image.
io.imsave('/users/wrdxo/desktop/apod_cropped.jpg', cropped)


-------------------------------------------------

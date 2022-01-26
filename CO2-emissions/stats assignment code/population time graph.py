"""
__author__= "Ward Dib" 
__date__= "2020-12-14"
__version__= "1"
__data_source__="shorturl.at/oPTW4"
"""

# Import libraries used in this code.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assign plotting style.
plt.style.use('bmh')

# Import all data files downloaded from The World Bank.
df1 = pd.read_csv('/Users/wrdxo/Desktop/Datasets/population.csv')
df2 = pd.read_csv('/Users/wrdxo/Desktop/Datasets/GHG_emissions.csv')
df3 = pd.read_csv('/Users/wrdxo/Desktop/Datasets/agriculture.csv')
df4 = pd.read_csv('/Users/wrdxo/Desktop/Datasets/fossile_energy.csv')
df5 = pd.read_csv('/Users/wrdxo/Desktop/Datasets/renewable_energy.csv')

# Create a subset of "population" containing the desired countries only.
population = df1[["Year", "United Kingdom", "Japan", "New Zealand", "World"]]
population.head()

# Create a subset of "ghg" containing the desired countries only.
ghg = df2[["Year", "United Kingdom", "Japan", "New Zealand", "World"]]
ghg.head()

# Normalise the new dataframes using min-max normalisation.
names = ['United Kingdom', 'Japan', 'New Zealand', 'World']

population[names] = population[names].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()))
ghg[names] = ghg[names].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()))

#print(population)
#print(ghg)

# Assign desired dates from dataframe to the x_axis as a numpy array.
x = population['Year']
x = x.to_numpy()

# Assign country columns from dataframe to the first y-axis.
y1 = population['United Kingdom']
y2 = population['Japan']
y3 = population['New Zealand']
y4 = population['World']
y5 = ghg['World']

# Create a figure and only one subplot.
fig, ax1 = plt.subplots(figsize = (15, 8))

# Create line plot for country populations.
ax1.plot(x, y1, label = 'UK', color = '#cb181d')
ax1.plot(x, y2, label = 'JPN', color = '#fd8d3c')
ax1.plot(x, y3, label = 'NZ', color = '#41ab5d')
ax1.plot(x, y4, label = 'World', color = '#4292c6')

# Creat scatter points for the greenhouse gas emissions in the world.
ax1.scatter(x, y5, marker = 'o', color = '#525252',
            label = 'World GhG Emissions')

# Add and format legend.
legend = ax1.legend(loc = "upper left", prop = {'size':16}, frameon = True,
                    framealpha = 1)
legend.get_frame().set_edgecolor('black')

# Fix tick formatting.
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1.tick_params(axis = 'both', which = 'minor', labelsize = 14)
ax1.xaxis.set_ticks(np.arange(1990, 2015, 2))

# Graph labels and format.
ax1.set_ylabel('Normalized Units', size = 14)
ax1.set_xlabel('Year', size = 14)
plt.grid(color = '#636363', linestyle = 'solid')
fig.patch.set_facecolor('white')
ax1.set_facecolor('#d9d9d9')

# Save figure to PC.
plt.savefig('fig3.png', dpi=600, bbox_inches='tight') 
plt.show()

#----------------------------------------------------------------------------
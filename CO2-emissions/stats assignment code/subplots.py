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
csfont = {'fontname':'Times New Roman'}

# Import all data files downloaded from The World Bank.
df1 = pd.read_csv('/Users/wrdxo/Desktop/Datasets/population.csv')
df2 = pd.read_csv('/Users/wrdxo/Desktop/Datasets/ghg%change.csv')
df3 = pd.read_csv('/Users/wrdxo/Desktop/Datasets/agriculture.csv')
df4 = pd.read_csv('/Users/wrdxo/Desktop/Datasets/fossile_energy.csv')
df5 = pd.read_csv('/Users/wrdxo/Desktop/Datasets/renewable_energy.csv')

# Create a subsets of the datasets containing the desired countries only.
population = df1[["Year", "United Kingdom", "Japan", "New Zealand", "World"]]
population.head()

ghg = df2[["Year", "United Kingdom", "Japan", "New Zealand", "World"]]
ghg.head()

agriculture = df3[["Year", "United Kingdom", "Japan", "New Zealand", "World"]]
agriculture.head()

fossilefuel = df4[["Year", "United Kingdom", "Japan", "New Zealand", "World"]]
fossilefuel.head()

renewablefuel = df5[["Year", "United Kingdom", "Japan",
                     "New Zealand", "World"]]
renewablefuel.head()

# Create a figure with 4 subplots that share a y-axis.
fig, axs = plt.subplots(2, 2, sharey = True)

# Adjust tick format.
plt.rcParams.update({"xtick.direction" : "in", 
                     "ytick.direction" : "in",
                     "figure.subplot.wspace" : 0.1,
                     "axes.xmargin" : 0,
                     "figure.figsize" : (15,8)})

# Assign desired dates from dataframe to the x_axis as a numpy array.
x = population['Year']
x = x.to_numpy()

# The first graph.
y1 = agriculture['World']
y2 = fossilefuel['World']
y3 = renewablefuel['World']
y4 = ghg['World']

l1, = axs[0, 0].plot(x, y1, color = '#41ab5d')
l2, = axs[0, 0].plot(x, y2, color = '#88419d')
l3, = axs[0, 0].plot(x, y3, color = '#0570b0')
l4, = axs[0, 0].plot(x, y4, color = '#525252', linewidth = 4,
                     linestyle = ':', alpha = 1)

axs[0, 0].set_title('World', weight = 'bold', **csfont)

# The second graph.
y1 = agriculture['United Kingdom']
y2 = fossilefuel['United Kingdom']
y3 = renewablefuel['United Kingdom']
y4 = ghg['United Kingdom']

l1, = axs[0, 1].plot(x, y1, color = '#41ab5d')
l2, = axs[0, 1].plot(x, y2, color = '#88419d')
l3, = axs[0, 1].plot(x, y3, color = '#0570b0')
l4, = axs[0, 1].plot(x, y4, color = '#525252', linewidth = 4,
                     linestyle = ':', alpha = 1)

axs[0, 1].set_title('United Kingdom', weight = 'bold', **csfont)

# The third graph.
y1 = agriculture['Japan']
y2 = fossilefuel['Japan']
y3 = renewablefuel['Japan']
y4 = ghg['Japan']

l1, = axs[1, 0].plot(x, y1, color = '#41ab5d')
l2, = axs[1, 0].plot(x, y2, color = '#88419d')
l3, = axs[1, 0].plot(x, y3, color = '#0570b0')
l4, = axs[1, 0].plot(x, y4, color = '#525252', linewidth = 4,
                     linestyle = ':', alpha = 1)

axs[1, 0].set_title('Japan', weight = 'bold', **csfont)

# The fourth graph.
y1 = agriculture['New Zealand']
y2 = fossilefuel['New Zealand']
y3 = renewablefuel['New Zealand']
y4 = ghg['New Zealand']

l1, = axs[1, 1].plot(x, y1, color = '#41ab5d')
l2, = axs[1, 1].plot(x, y2, color = '#88419d')
l3, = axs[1, 1].plot(x, y3, color = '#0570b0')
l4, = axs[1, 1].plot(x, y4, color = '#525252', linewidth = 4,
                     linestyle = ':', alpha = 1)

axs[1, 1].set_title('New Zealand', weight = 'bold', **csfont)

# Format ticks, labels, and colours.
for ax in axs.flat:
    ticklabels = ax.get_xticklabels()
    ticklabels[0].set_ha("left")
    ticklabels[-1].set_ha("right")
    ax.set_ylabel('Usage %', size = 14)
    ax.set_xlabel('Year', size = 14)
    ax.label_outer()
    ax.grid()
    ax.xaxis.set_ticks(np.arange(1992, 2013, 5))
    ax.yaxis.set_ticks(np.arange(0, 100, 20))
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 14)
    ax.grid(color = '#636363', linestyle = 'solid')
    ax.set_facecolor('#d9d9d9')

# Format legend.
l1.set_label('Agriculture')
l2.set_label('Fossile Fuel')
l3.set_label('Renewable Energy')
l4.set_label('GhG emissions')
  
line_labels = ["Agriculture", "Fossile Fuel",
               "Renewable Energy", "GhG emissions"] 

fig.legend([l1, l2, l3, l4], labels = line_labels, loc = "upper left", 
           borderaxespad = 0.1, bbox_to_anchor=(0.05, 1.01), prop = {'size':16},
           frameon = True, framealpha = 1, edgecolor='black')
plt.subplots_adjust(right = 0.8)

# Save figure to PC.
plt.savefig('fig2.png', dpi=600, bbox_inches='tight') 
plt.show()

#----------------------------------------------------------------------------
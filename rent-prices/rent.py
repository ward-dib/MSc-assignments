"""
__author__= "Ward Dib" 
__date__= "2020-11-27"
__version__= "1"
__data_source__="1. https://www.ons.gov.uk/employmentandlabourmarket/
                    peopleinwork/earningsandworkinghours/datasets/
                    ashe1997to2015selectedestimates
                    2.https://landregistry.data.gov.uk/app/ukhpi/browse?
                    from=2000-01-01&location=http%3A%2F%2Flandregistry.data.
                    gov.uk%2Fid%2Fregion%2Fengland&to=2020-10-01&lang=en"

"""

# Import libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# GRAPH 1.


# Fetch dataframe from downloaded csv file.
price = pd.read_csv('/Users/wrdxo/Desktop/wrd/Uni/MSc/Applied Data Science 1/'
                    'Assignment 1/datasets/price_vs_sales_2000_to_2020.csv')

# Change plot style.
# print(plt.style.available)
plt.style.use('seaborn')

# To only operate on elements beginning in January, we assign a new dataframe.
years_list = (price[price['Pivotable date'].str.match('Jan')])
# print(years_list)

# Assign desired dates from dataframe to the x_axis as a numpy array.
x_array = years_list['Pivotable date']
x_array = x_array.to_numpy()

# Assign price column from dataframe to the first y-axis as a numpy array.
y1_array = years_list['Average price All property types']
y1_array = y1_array.to_numpy()

# Assign sales column from dataframe to the second y-axis as a numpy array.
y2_array = years_list['Sales volume']
y2_array = y2_array.to_numpy()

# Create a figure and only one subplot.
fig, ax1 = plt.subplots(figsize = (15, 8))

 # Instantiate a second axes that shares the same x-axis.
ax2 = ax1.twinx() 

# Create line plot.
ax1.plot(x_array, y1_array, color = '#f768a1')

# Add a second y-axis to the opposite side.
ax2.plot(x_array, y2_array, color = '#6a51a3')

# Change tick names to exclude the word 'Jan'.
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
            20, 21], ["2000", "2001", "2002", "2003", "2004", "2005", 
                          "2006", "2007", "2008", "2009", "2010", "2011", 
                          "2012", "2013", "2014", "2015", "2016", "2017", 
                          "2018", "2019", "2020"])

# Add a legend to help interpret the plot.
colors = {'Price average':'#f768a1',
          'Sales volume':'#6a51a3'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color = colors[label]) for label
           in labels]
plt.legend(handles, labels, labelcolor = 'linecolor', loc = 'upper left',
           fontsize = "x-large")
plt.title('Figure 2', weight = 'bold', y = -0.13)

# Format ticks to include the "£" sign and commas every 1000s for viewing ease.
fmt = '£{x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax1.yaxis.set_major_formatter(tick)

fmt = '{x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax2.yaxis.set_major_formatter(tick)
plt.yticks(np.arange(0, max(y1_array), 25000))

# Function used to rotate ticks and right align them, does not return values.
fig.autofmt_xdate()

# Save figure to PC.
plt.savefig('fig2.png', dpi=300, bbox_inches='tight') 

plt.show()

# ---------------------------------------------------------------------------

# GRAPH 2.

# Import libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Fetch dataframe from downloaded csv file.
df = pd.read_csv('/Users/wrdxo/Desktop/wrd/Uni/MSc/Applied Data Science 1/'
                 'Assignment 1/datasets/income-avg.csv')

# Change plot style.
plt.style.use('seaborn')

# Append year column from csv file into a list.
years_list = df['Pivotable date']
print(years_list)

# Assign desired dates from dataframe to the x_axis as a numpy array.
x_array = df['Pivotable date']
x_array = x_array.to_numpy()

# Assign price and income columns from dataframe to the y-axis.
y1_array = df['Average price All property types']
y2_array = df['Average annual income in England']

# Create a figure and only one subplot.
fig, ax = plt.subplots(figsize = (15, 8))

# Creat line plot and shade in the area underneath the line.
ax.plot(x_array, y1_array, color = '#f768a1')
ax.plot(x_array, y2_array, color = '#6a51a3')
ax.fill_between(x_array, y1_array, color = '#f768a1', alpha = 0.12)
ax.fill_between(x_array, y2_array, color = '#bcbddc', alpha = 0.4)
             
# Format ticks.
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
            20, 21], ["2000", "2001", "2002", "2003", "2004", "2005", 
                          "2006", "2007", "2008", "2009", "2010", "2011", 
                          "2012", "2013", "2014", "2015", "2016", "2017", 
                          "2018", "2019", "2020"])

# Add a legend to help interpret the plot.
colors = {'Price':'#f768a1',
          'Income':'#6a51a3'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color = colors[label]) for label
           in labels]
plt.legend(handles, labels, labelcolor = 'linecolor', loc = 'upper left',
           fontsize = "x-large")
plt.title('Figure 1', weight = 'bold', y = -0.13)

# Format ticks.
fmt = '£{x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)

ax.yaxis.set_major_formatter(tick) 
plt.yticks(np.arange(0, max(y1_array), 20000))

fig.autofmt_xdate()

# Save figure to PC.
plt.savefig('fig1.png', dpi = 300, bbox_inches = 'tight')

plt.show()

# ---------------------------------------------------------------------------

# GRAPH 3.


# Fetch dataframes from downloaded csv file.
price = pd.read_csv('C:/Users/wrdxo/Desktop/wrd/Uni/MSc/Applied Data Science '
                    '1/Assignment 1/datasets/price_vs_sales_2000_to_2020.csv')

income = pd.read_csv('C:/Users/wrdxo/Desktop/wrd/Uni/MSc/Applied Data Science '
                     '1/Assignment 1/datasets/salary_1997_2020_timeseries.csv')

# Change plot style.
plt.style.use('seaborn')

# Slice the dataframe to get only desired elements.
price = price.iloc[[0, 240], [6]]
income = income.iloc[[6], [5, 25]]
#print(price)
#print(income)

'''
     Average price All property types
0                               75219
240                            248386
   Unnamed: 5 Unnamed: 25
6     19107.0       31777

'''
 
# From the print(df) command, get values of each group.
bars1 = [19107, 31777]
bars2 = [75219, 248386]
 
# Find total heights of bars1 + bars2.
bars = np.add(bars1, bars2).tolist()
 
# Choose position of bars on the x-axis.
r = [0,1]
 
# Name the groups.
names = ['2000','2020']

# Create a figure and only one subplot.
fig, ax = plt.subplots(figsize = (5, 8))

# Create stacked bar plot.
plt.bar(r, bars1, color = '#807dba', edgecolor = 'black', width = 0.5,
        alpha = 0.6)
plt.bar(r, bars2, bottom = bars1, color = '#f768a1', edgecolor = 'black', 
        width = 0.5, alpha = 0.7)
colors = {'income':'#807dba', 'property price':'#f768a1'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color = colors[label]) for label
           in labels]
plt.legend(handles, labels, fontsize = "large")
plt.title('Figure 3', weight = 'bold', y = -0.07)

# Format ticks.
plt.xticks(r, names, fontweight = 'bold')

fmt = '£{x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick) 

# Save figure to PC.
plt.savefig('fig3.png', dpi = 300, bbox_inches = 'tight')

plt.show()

# ---------------------------------------------------------------------------

# GRAPH 4 (not used in report).

# Import libraries.
import pandas as pd
import matplotlib.pyplot as plt

# Fetch dataframes from downloaded csv file.
rent = pd.read_csv('C:/Users/wrdxo/Desktop/wrd/Uni/MSc/Applied Data Science '
                   '1/Assignment 1/datasets/week_price_private_rent.csv')

income = pd.read_csv('C:/Users/wrdxo/Desktop/wrd/Uni/MSc/Applied Data Science '
                     '1/Assignment 1/datasets/salary_1997_2020_timeseries.csv')

# Change plot style.
plt.style.use('seaborn')

# Slice dataframe and only append the needed columns and rows to a numpy array.
rent = rent.iloc[4, 5:26]
rent = rent.to_numpy()
income = income.iloc[24, 5:26]
income = income.to_numpy()
#print(rent)
#print(income)

# Assign x-axis range manually.
x = range(0,21)

# From print(df) commands, fetch needed values for y-axes.
y1 = [51.92, 53.11, 53.9, 55.81, 56.52, 58.23, 61.49, 64.32, 66.67,
      69.96, 73.51, 77.91, 78.28, 83.20875366, 88.40994266, 92.3,
      95.89, 97.83768054, 96.61112068, 95.59007351, 95.11734937]
y2 = [364.4, 381.7, 396.5, 410.6, 425.0, 436.0, 452.3, 463.6, 483.9, 495.0, 
      504.5, 507.2, 512.6, 520.3, 523.5, 531.6, 544.2, 555.8, 574.8, 592.2,
      589.9]

# Create an area stack graph.
plt.figure(figsize = (10, 10))
pal = ["#f768a1", "#9e9ac8"]
plt.stackplot(x, y1, y2, labels = ['Rent', 'Income'], colors = pal,
              alpha = 0.5)
plt.title('Average rent vs. Average income (per week)', weight = 'bold')
plt.ylabel('Income (£)')
plt.legend(loc = 'upper left', fontsize = "x-large")
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
            20, 21, 22], ["2000", "2001", "2002", "2003", "2004", "2005", 
                          "2006", "2007", "2008", "2009", "2010", "2011", 
                          "2012", "2013", "2014", "2015", "2016", "2017", 
                          "2018", "2019", "2020", "2021"], rotation = 45)

# Save figure to PC.
plt.savefig('fig3.png', dpi = 300, bbox_inches = 'tight')

plt.show()

# ---------------------------------   END  -----------------------------------
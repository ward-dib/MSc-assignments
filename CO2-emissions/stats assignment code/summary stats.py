"""
__author__= "Ward Dib" 
__date__= "2020-12-14"
__version__= "1"
__data_source__="shorturl.at/oPTW4"
"""

# Import libraries used in this code.
import pandas as pd

# Get values from 1992 and 2012 for all datasets and all indicators.
# Population.
col_list = ["United Kingdom", "Japan", "New Zealand", "World"]
df1 = pd.read_csv("/Users/wrdxo/Desktop/Datasets/population.csv",
                 usecols=col_list, skiprows = range(2, 21))
print(df1)

# Calculate percentag increase or decrease from 1992 up until 2012.
percentage_change = df1.pct_change()
print(percentage_change)

# Greenhouse gas emissions.
col_list = ["United Kingdom", "Japan", "New Zealand", "World"]
df2 = pd.read_csv("/Users/wrdxo/Desktop/Datasets/GHG_emissions.csv",
                 usecols=col_list, skiprows = range(2, 21))
print(df2)
percentage_change = df2.pct_change()
print(percentage_change)

# Agriculture land (% of land area).
col_list = ["United Kingdom", "Japan", "New Zealand", "World"]
df3 = pd.read_csv("/Users/wrdxo/Desktop/Datasets/agriculture.csv",
                 usecols=col_list, skiprows = range(2, 21))
print(df3)
percentage_change = df3.pct_change()
print(percentage_change)

# Fossile fuel energy consumption (% of total).
col_list = ["United Kingdom", "Japan", "New Zealand", "World"]
df4 = pd.read_csv("/Users/wrdxo/Desktop/Datasets/fossile_energy.csv",
                 usecols=col_list, skiprows = range(2, 21))

print(df4)
percentage_change = df4.pct_change()
print(percentage_change)

# Renewable energy consumption (% of total).
col_list = ["United Kingdom", "Japan", "New Zealand", "World"]
df5 = pd.read_csv("/Users/wrdxo/Desktop/Datasets/renewable_energy.csv",
                 usecols = col_list, skiprows = range(2, 21))
print(df5)
percentage_change = df5.pct_change()
print(percentage_change)

#----------------------------------------------------------------------------
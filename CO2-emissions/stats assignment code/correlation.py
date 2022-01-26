"""
__author__= "Ward Dib" 
__date__= "2020-12-14"
__version__= "1"
__data_source__="shorturl.at/oPTW4"
"""

# Import libraries used in this code.
import pandas as pd
from scipy import stats

# Import all data files downloaded from The World Bank.
df1 = pd.read_csv('/Users/wrdxo/Desktop/Datasets/population.csv')
df2 = pd.read_csv('/Users/wrdxo/Desktop/Datasets/GHG_emissions.csv')
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


# Summary statistics.

describe1 = population[["United Kingdom", "Japan",
                       "New Zealand", "World"]].describe()
describe2 = ghg[["United Kingdom", "Japan",
                "New Zealand", "World"]].describe()
describe3 = agriculture[["United Kingdom", "Japan",
                        "New Zealand", "World"]].describe()
describe4 = fossilefuel[["United Kingdom", "Japan",
                        "New Zealand", "World"]].describe()
describe5 = renewablefuel[["United Kingdom", "Japan",
                          "New Zealand", "World"]].describe()

print(describe1)
print(describe2)
print(describe3)
print(describe4)
print(describe5)

# Correlation between GhG emissions and climate indicators: World data.
x = ghg['World']
x = x.to_numpy()
y1 = population['World']
y1 = y1.to_numpy()
y2 = fossilefuel['World']
y2 = y2.to_numpy()
y3 = renewablefuel['World']
y3 = y3.to_numpy()
y4 = agriculture['World']
y4 = y4.to_numpy()

# P-value & Pearson’s correlation coefficient.
r, p = stats.pearsonr(x, y1)
r2, p2 = stats.pearsonr(x, y2)
r3, p3 = stats.pearsonr(x, y3)
r4, p4 = stats.pearsonr(x, y4)

# Round to 3 decimal places.
r = round(r, 3)
p = round(p, 3)

r2 = round(r2, 3)
p2 = round(p2, 3)

r3 = round(r3, 3)
p3 = round(p3, 3)

r4 = round(r4, 3)
p4 = round(p4, 3)

# Print all values, with conditions.
print("World: GhG & population")
print(r, p)
if (r < 0):
    print("negative correlation")
if (0 < r < 1):
    print("positive correlation")
if (r == 0):
    print("no correlation")

if (p <= 0.05):
    print("statistically significant") 
else:
    print("not statistically significant")
print("-----------------------------")

print("World: GhG & Fossile Fuel")  
print(r2, p2)
if (r < 0):
    print("negative correlation")
if (0 < r < 1):
    print("positive correlation")
if (r == 0):
    print("no correlation")

if (p <= 0.05):
    print("statistically significant") 
else:
    print("not statistically significant")
print("-----------------------------")

print("World: GhG & Renewable Energy")   
print(r3, p3)
if (r < 0):
    print("negative correlation")
if (0 < r < 1):
    print("positive correlation")
if (r == 0):
    print("no correlation")

if (p <= 0.05):
    print("statistically significant") 
else:
    print("not statistically significant")
print("-----------------------------")

print("World: GhG & Agriculture")   
print(r4, p4)
if (r < 0):
    print("negative correlation")
if (0 < r < 1):
    print("positive correlation")
if (r == 0):
    print("no correlation")

if (p <= 0.05):
    print("statistically significant") 
else:
    print("not statistically significant")
print("-----------------------------")  

# Correlation between GhG emissions and climate indicators: UK data.
x = ghg['United Kingdom']
x = x.to_numpy()
y1 = population['United Kingdom']
y1 = y1.to_numpy()
y2 = fossilefuel['United Kingdom']
y2 = y2.to_numpy()
y3 = renewablefuel['United Kingdom']
y3 = y3.to_numpy()
y4 = agriculture['United Kingdom']
y4 = y4.to_numpy()


r, p = stats.pearsonr(x, y1)
r2, p2 = stats.pearsonr(x, y2)
r3, p3 = stats.pearsonr(x, y3)
r4, p4 = stats.pearsonr(x, y4)

r = round(r, 3)
p = round(p, 3)

r2 = round(r2, 3)
p2 = round(p2, 3)

r3 = round(r3, 3)
p3 = round(p3, 3)

r4 = round(r4, 3)
p4 = round(p4, 3)

print("United Kingdom: GhG & population")
print(r, p)
if (r < 0):
    print("negative correlation")
if (0 < r < 1):
    print("positive correlation")
if (r == 0):
    print("no correlation")

if (p <= 0.05):
    print("statistically significant") 
else:
    print("not statistically significant")
print("-----------------------------")

print("United Kingdom: GhG & Fossile Fuel")  
print(r2, p2)
if (r < 0):
    print("negative correlation")
if (0 < r < 1):
    print("positive correlation")
if (r == 0):
    print("no correlation")

if (p <= 0.05):
    print("statistically significant") 
else:
    print("not statistically significant")
print("-----------------------------")

print("United Kingdom: GhG & Renewable Energy")   
print(r3, p3)
if (r < 0):
    print("negative correlation")
if (0 < r < 1):
    print("positive correlation")
if (r == 0):
    print("no correlation")

if (p <= 0.05):
    print("statistically significant") 
else:
    print("not statistically significant")
print("-----------------------------")  

print("United Kingdom: GhG & Agriculture")   
print(r4, p4)
if (r < 0):
    print("negative correlation")
if (0 < r < 1):
    print("positive correlation")
if (r == 0):
    print("no correlation")

if (p <= 0.05):
    print("statistically significant") 
else:
    print("not statistically significant")
print("-----------------------------") 

# Correlation between GhG emissions and climate indicators: JPN data.
x = ghg['Japan']
x = x.to_numpy()
y1 = population['Japan']
y1 = y1.to_numpy()
y2 = fossilefuel['Japan']
y2 = y2.to_numpy()
y3 = renewablefuel['Japan']
y3 = y3.to_numpy()
y4 = agriculture['Japan']
y4 = y4.to_numpy()


r, p = stats.pearsonr(x, y1)
r2, p2 = stats.pearsonr(x, y2)
r3, p3 = stats.pearsonr(x, y3)
r4, p4 = stats.pearsonr(x, y4)

# Round to 3 decimal places.
r = round(r, 3)
p = round(p, 3)

r2 = round(r2, 3)
p2 = round(p2, 3)

r3 = round(r3, 3)
p3 = round(p3, 3)

r4 = round(r4, 3)
p4 = round(p4, 3)

print("Japan: GhG & population")
print(r, p)
if (r < 0):
    print("negative correlation")
if (0 < r < 1):
    print("positive correlation")
if (r == 0):
    print("no correlation")

if (p <= 0.05):
    print("statistically significant") 
else:
    print("not statistically significant")  
print("-----------------------------")

print("Japan: GhG & Fossile Fuel")  
print(r2, p2)
if (r < 0):
    print("negative correlation")
if (0 < r < 1):
    print("positive correlation")
if (r == 0):
    print("no correlation")

if (p <= 0.05):
    print("statistically significant") 
else:
    print("not statistically significant")  
print("-----------------------------")

print("Japan: GhG & Renewable Energy")   
print(r3, p3)
if (r < 0):
    print("negative correlation")
if (0 < r < 1):
    print("positive correlation")
if (r == 0):
    print("no correlation")

if (p <= 0.05):
    print("statistically significant") 
else:
    print("not statistically significant")
print("-----------------------------")  

print("Japan: GhG & Agriculture")   
print(r4, p4)
if (r < 0):
    print("negative correlation")
if (0 < r < 1):
    print("positive correlation")
if (r == 0):
    print("no correlation")

if (p <= 0.05):
    print("statistically significant") 
else:
    print("not statistically significant")
print("-----------------------------") 

# Correlation between GhG emissions and climate indicators: NZ data.
x = ghg['New Zealand']
x = x.to_numpy()
y1 = population['New Zealand']
y1 = y1.to_numpy()
y2 = fossilefuel['New Zealand']
y2 = y2.to_numpy()
y3 = renewablefuel['New Zealand']
y3 = y3.to_numpy()
y4 = agriculture['New Zealand']
y4 = y4.to_numpy()


r, p = stats.pearsonr(x, y1)
r2, p2 = stats.pearsonr(x, y2)
r3, p3 = stats.pearsonr(x, y3)
r4, p4 = stats.pearsonr(x, y4)

r = round(r, 3)
p = round(p, 3)

r2 = round(r2, 3)
p2 = round(p2, 3)

r3 = round(r3, 3)
p3 = round(p3, 3)

r4 = round(r4, 3)
p4 = round(p4, 3)

print("New Zealand: GhG & population")
print(r, p)
if (r < 0):
    print("negative correlation")
if (0 < r < 1):
    print("positive correlation")
if (r == 0):
    print("no correlation")

if (p <= 0.05):
    print("statistically significant") 
else:
    print("not statistically significant")   
print("-----------------------------")

print("New Zealand: GhG & Fossile Fuel")  
print(r2, p2)
if (r < 0):
    print("negative correlation")
if (0 < r < 1):
    print("positive correlation")
if (r == 0):
    print("no correlation")

if (p <= 0.05):
    print("statistically significant") 
else:
    print("not statistically significant")
print("-----------------------------")

print("New Zealand: GhG & Renewable Energy")   
print(r3, p3)
if (r < 0):
    print("negative correlation")
if (0 < r < 1):
    print("positive correlation")
if (r == 0):
    print("no correlation")

if (p <= 0.05):
    print("statistically significant") 
else:
    print("not statistically significant")
print("-----------------------------")

print("New Zealand: GhG & Agriculture")   
print(r4, p4)
if (r < 0):
    print("negative correlation")
if (0 < r < 1):
    print("positive correlation")
if (r == 0):
    print("no correlation")

if (p <= 0.05):
    print("statistically significant") 
else:
    print("not statistically significant")
print("-----------------------------") 

"""
Results:
    
World: GhG & population
0.93925 0.0
positive correlation
statistically significant
-----------------------------
World: GhG & Fossile Fuel
0.91585 0.0
positive correlation
statistically significant
-----------------------------
World: GhG & Renewable Energy
-0.68087 0.00068
positive correlation
statistically significant
-----------------------------
World: GhG & Agriculture
0.11443 0.62139
positive correlation
statistically significant
-----------------------------
United Kingdom: GhG & population
-0.955 0.0
negative correlation
statistically significant
-----------------------------
United Kingdom: GhG & Fossile Fuel
0.252 0.27
negative correlation
statistically significant
-----------------------------
United Kingdom: GhG & Renewable Energy
-0.853 0.0
negative correlation
statistically significant
-----------------------------
United Kingdom: GhG & Agriculture
0.37505 0.09389
negative correlation
statistically significant
-----------------------------
Japan: GhG & population
0.311 0.17
positive correlation
not statistically significant
-----------------------------
Japan: GhG & Fossile Fuel
0.354 0.115
positive correlation
not statistically significant
-----------------------------
Japan: GhG & Renewable Energy
-0.23 0.315
positive correlation
not statistically significant
-----------------------------
Japan: GhG & Agriculture
-0.27404 0.22934
positive correlation
not statistically significant
-----------------------------
New Zealand: GhG & population
0.787 0.0
positive correlation
statistically significant
-----------------------------
New Zealand: GhG & Fossile Fuel
-0.208 0.366
positive correlation
statistically significant
-----------------------------
New Zealand: GhG & Renewable Energy
-0.001 0.997
positive correlation
statistically significant
-----------------------------
New Zealand: GhG & Agriculture
-0.83857 0.0
positive correlation
statistically significant
-----------------------------
"""

#----------------------------------------------------------------------------
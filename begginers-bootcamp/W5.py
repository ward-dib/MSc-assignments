"""
Created on Thu Dec 10 09:05:01 2020

@author: wrddib

Weekly challenge W5

– Produce summary statistics and plots for the price, accommodation capacity
and availability of AirBnBs.

– Calculate Kendall’s τ for the review scores for location and value for money. 
What do youinterpret from the result?

– Examine the review scores for value for money for accommodation with
prices in excess of £1000 and less then £100. Perform a KS-test and
interpret your result.
    
"""

# Import libraries needed.
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Fetch dataframe from downloaded csv file.
df = pd.read_csv('/Users/wrdxo/Downloads/listings.csv')

# Assign a subset with only the needed columns.
listings = df[["accommodates", "availability_365",  "price",
               "review_scores_location", "review_scores_value"]]
listings.head()

# Summary stats.
print(listings.describe())
print(listings.median())
print(listings.mean())

# Assign each column as a variable for plotting and for correlation tests.
a = listings['accommodates']
a = a.to_numpy()

b = listings['price']
b = b.to_numpy()

c = listings['availability_365']
c = c.to_numpy()

d = listings['review_scores_location']
d = d.to_numpy()

e = listings['review_scores_value']
e = e.to_numpy()

# Plot a scatter graph and change variables as needed.
fig, ax = plt.subplots(figsize = (15, 8))
ax.scatter(a, b)
plt.show()

# Kendall rank correlation coefficient.
t, p = stats.kendalltau(d, e)

print(t, p)

"""

The Kendall correlation coefficient between the reviews for location and value
for money is τ = 0.742, with a p-value = 0.00.
This is close to 1, so it indicates correlation between the variables.
The p-value means that you should reject the null hypothesis, that being
that the variables are not correlated. So, from the two values we conclude
that there's a correlation between location and value for money for an airbnb 
listing.

"""

# KS-test for listings less then £100.

b = range(0, 100)
d, p = stats.ks_2samp(b, e)
print(d, p)

# KS-test for listings more then £1000.

b = range(1000, 20000)
d, p = stats.ks_2samp(b, e)
print(d, p)

"""

The KS-test assesses the independence of two variables.
It was done over two ranges, between the reviews for price
and value for money.

For the first range, for airbnb listing costing <£100:
D = 0.89 and p-value = 0.00.

The D value is high and the p-value close to zero,
so we can reject the null hypothesis that the two variables are drawn from
the same distribution.

For the second range, for listings costing >£1000.
D = 1.0 and p-value = 0.00.

Again, the D value is the highest it can be and the p-value is zero.
We can reject the null hypothesis that the two variables are drawn from
the same distribution.

"""
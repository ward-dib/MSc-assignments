import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn import preprocessing

#------------------------------- T A S K 1 ------------------------------------

# a. Use Pandas to load the data and report the number of data points (rows) in the dataset.

wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep = ';')
wine.head()

print(f'The white wine dataset has {len(wine)} data points.')
print(wine.shape)

#-----------------------------------------------------------------------------
# b. Consider “quality” as class labels. Report the number of features in the dataset and the number of data points in each class.

print('Number of columns in dataset =', len(wine.columns))
features = len(wine.columns) - 1
print('Number of features in dataset =', features)
wine.groupby(['quality']).agg(['count'])
count_per_class = wine.groupby('quality')['quality'].count()
print("Number of data points per class:\n", count_per_class)
count_per_class.sum(axis = 0)
count_table = count_per_class.to_frame()
count_table = count_table.rename(columns={'quality': 'data points in each class'})
print(count_table)

#-----------------------------------------------------------------------------

# c. Perform random permutations of the data using the function, shuffle, from sklearn.utils. 

wine_shuffle = shuffle(wine, random_state = 20)
wine_shuffle = wine_shuffle.reset_index(drop = True)

print(wine_shuffle)

#-----------------------------------------------------------------------------

# d. Produce one scatter plot, that is, one feature against another feature. 

# Creating figure
fig = plt.figure()
ax = plt.gca()

ax.scatter('fixed acidity', 'density', data = wine, color = 'b', s = 8)

ax.set_title("Fixed Acidity vs. Density")

#------------------------------- T A S K 2 ------------------------------------

# a. Perform a PCA analysis on the whole white_wine dataset.

pca_analysis = PCA(n_components = 2)
wine_pca = pca_analysis.fit_transform(wine)

print(wine_pca)

#-----------------------------------------------------------------------------

# b. Plot the data in the PC1 and PC2 projections and label/colour the data in the plot according to their class labels.

x = StandardScaler().fit_transform(wine)
wine_pca = pca_analysis.fit_transform(x)

PC1 = wine_pca[:, 0]
PC2 = wine_pca[:, 1]

fig, ax = plt.subplots()

scatter = ax.scatter(PC1, PC2, c = wine['quality'])

# produce a legend with the unique colors from the scatter
legend = ax.legend(*scatter.legend_elements(),
                    loc = "upper right", title = "Classes")
ax.add_artist(legend)

plt.xlabel('PC1', weight = 'bold')
plt.ylabel('PC2', weight = 'bold')
plt.show

#-----------------------------------------------------------------------------

# c. Report the variance captured by each principal component. 

C = wine[['fixed acidity', 'volatile acidity', 'citric acid',
                'residual sugar','chlorides', 'free sulfur dioxide',
                'total sulfur dioxide', 'density', 'pH', 'sulphates',
                'alcohol']]

features = C.T

covariance_matrix =  np.cov(features)

print(covariance_matrix)

#------------------------------- T A S K 3 ------------------------------------

# a. Take out the first 1000 rows from white_wine and save it as the validation set.
# b. Take out the last 1000 rows from white_wine and save it as the test set.
# c. Save the rest of rows from white_wine as the training set.

features = wine[['fixed acidity', 'volatile acidity',
                 'citric acid', 'residual sugar','chlorides',
                 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']]

target = wine[['quality']]

print(wine.shape)

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size = (500/2449),
                                                    shuffle = False)

print(X_test.shape)
print(X_train.shape)

X_trn, X_val, y_trn, y_val = train_test_split(X_train,
                                              y_train,
                                              test_size = (1000/3898),
                                              shuffle = False)
print(X_trn.shape)
print(X_val.shape)
print(y_trn.shape)
print(y_val.shape)

#------------------------------- T A S K 4 ------------------------------------
"""
a. Produce a learning curve of the size of training set against the performance
measurements. The performance should be measured on both the training set and the
validation set. You need to choose at least 10 different sizes for the training set. For
example, the first size may be 10% of the total training set produced in Task 3.
"""        

train_sizes = [1, 289, 579, 869, 1159, 1449, 1738, 2028, 2318, 2608, 2898]
#train_sizes = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#train_sizes = [1, 500, 1500, 2000, 2500, 3000, 3500, 3918]

features = ['fixed acidity', 'volatile acidity',
            'citric acid', 'residual sugar','chlorides',
            'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']
target = ['quality']

train_sizes, train_scores, validation_scores = learning_curve( 
    estimator = LinearRegression(),
    X = wine[features],
    y = wine[target], train_sizes = train_sizes, cv = 5,
    scoring = 'neg_mean_squared_error')

train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)

plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('RMSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for white wine linear regression', fontsize = 16)
plt.legend()

#-----------------------------------------------------------------------------
# b. Report what the best training data size you would like to use for this work is and explain why you choose it.

"""
The best training size is the following. 
This training size decreases bias and increase variance.
"""

best_train_size = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

#-----------------------------------------------------------------------------
# c. Report the performance on the test set obtained using the model trained from the best size.

"""
When running the code above on the best training size, we can see from the plot that it gives a larger
gap between the errors. A narrow gap indicates low variance, and a wider gap
indicates greater the variance. It also gives a low training error and 
a higher validation error, which is the norm.
"""

#------------------------------- T A S K 5 ------------------------------------

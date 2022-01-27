import os
import sys
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import  Sequential
from keras.layers import  Dense, Activation
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
import matplotlib.pyplot as plt
import statistics

df = pd.read_excel(r'C:\Users\warda\Desktop\wrd\Uni\MSc\Project\C19.xlsx')
df.head()
df.drop(['Sex2'], axis = 1, inplace = True)
df.drop(['ID'], axis = 1, inplace = True)
df.drop(['ICU'], axis = 1, inplace = True)
df.drop(['Days'], axis = 1, inplace = True)
df.head()
dataset = df.values
dataset
X = dataset[:,0:8]
Y = dataset[:,8]
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_scale
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.5)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

model = Sequential([
    Dense(32, activation='relu', input_shape=(8,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist = model.fit(X_train, Y_train,
          batch_size=32, epochs=2000,
          validation_data=(X_val, Y_val))
model.evaluate(X_test, Y_test)[1]

Y_pred = model.predict(X_test).ravel()

fpr, tpr, threshold = roc_curve(Y_test, Y_pred)
roc_auc = auc(fpr, tpr)

plt.rcParams["figure.figsize"] = (10.66, 10.91)
#plt.plot([-1, 0], [0, 1], color='grey', linestyle='-', alpha=0.3)
plt.plot(fpr-1, tpr, color="#e6550d",linewidth=2.2, label='Neural Network' % roc_auc)


#plt.savefig("ANNROC.png", transparent=True, dpi=300)
roc_auc
import numpy as np
from scipy.stats import sem
from sklearn.metrics import roc_auc_score
n_bootstraps = 2000
rng_seed = 142  # control reproducibility
bootstrapped_scores = []

rng = np.random.RandomState(rng_seed)
for i in range(n_bootstraps):
    indices = rng.randint(0, len(Y_pred), len(Y_pred))
    if len(np.unique(Y_test[indices])) < 2:
        continue
    score = roc_auc_score(Y_test[indices], Y_pred[indices])
    bootstrapped_scores.append(score)
    print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    import matplotlib.pyplot as plt
plt.hist(bootstrapped_scores, bins=50)
plt.title('Histogram of the bootstrapped ROC AUC scores')
plt.show()
sorted_scores = np.array(bootstrapped_scores)
sorted_scores.sort()

confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
print("Confidence interval : [{:0.3f} - {:0.3}]".format(
    confidence_lower, confidence_upper))

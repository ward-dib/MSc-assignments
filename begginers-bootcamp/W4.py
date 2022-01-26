# Simple visualisation W4
# the best way to communicate data by plots
# matplotlib plotting module is the most widely used
# matplotlib documentation has all the arguments for shapes and colours

'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

print(mpl.matplotlib_fname())

print(plt.style.available)
plt.style.use('fivethirtyeight')
plt.style.use('ggplot')
# plt.style.use('dark_background')

# try to come up with ur own style as part of the excercise 

x = np.linspace(-np.pi, np.pi, 100)

plt.figure(0, figsize=(5,5)) # to change size, write tuple in inches

plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)', linestyle='dashed')



plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('my plot')

noisy= np.sin(x) + np.random.normal (0, 0, len(x))
plt.plot(x, noisy, label='noisy sin(x)', linestyle='none', marker='o',
         markerfacecolor='blue', markeredgecolor='black')

plt.legend()

# new figure

plt.figure(1, figsize=(10,5))

sample=np.random.normal(0,2,1000)
sample2=np.random.normal(2,2,100)

plt.subplots_adjust(hspace=0.4, wspace=0.4) #adjusts space between subplots

plt.subplot(1,2,1) # indextig starts at 1 here
plt.hist(sample, bins=50, label='sample1')

plt.xlabel('sample')
plt.ylabel('number')

plt.subplot(1,2,2)
plt.hist(sample2,alpha=0.5, label='sample2')

plt.xlabel('sample')
plt.ylabel('number')
plt.title('my plot')
# plt.savefig('fig1.png') to save '''


# ------------------------------------------------------------------------
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# import matplotlib as mpl

sample1 = np.random.normal(5,2,100)
sample2 = np.random.normal(0,10,100)

sample = np.concatenate((sample1, sample2))

plt.hist(sample, density=True)

kde = stats.gaussian_kde(sample, bw_method=0.2)

xs = np.linspace(np.min(sample), np.max(sample), 100)

plt.plot(xs, kde(xs), linewidth=4)

# function that's gonna return for a given value on the x axis what the kernel 
# estimate density is

'''
# ------------------------------------------------------------------------

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sample1 = np.random.normal(5,2,100)
sample2 = np.random.normal(0,10,100)

fig, ax = plt.subplots(1,1)

ax.violinplot([sample1,sample2], showmedians=True)

ax.set_xticks([1,2])
ax.set_xticklabels=(['sample 1','sample 2'])

# plt.boxplot([sample1, sample2], labels=['sample 1', 'sample 2'])
# called box and whisker plot

'''

# ------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


df= pandas.read_excel('')

fig, ax=plt.subplots(1,1)

#df.boxplot(column='rain', by='yyyy', ax=ax)
df.boxplot(column='rain', by='mm', ax=ax)
# u can explore the data graphically by showing it in different ways
ax.set_xlabel('')

# ------------------------------------------------------------------------
























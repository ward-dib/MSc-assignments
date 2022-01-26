# Importing needed libraries.
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Import the dataset.
df = pd.read_csv('/Users/wrdxo/Desktop/wrd/Uni/MSc/Applied Data Science 1/Assignment 3/anime dataset.csv')

# Change plot style.
print(plt.style.available)
plt.style.use('seaborn')
plt.rcParams['axes.facecolor'] = '#DEEBF7'

# Performing some summary statistics to examin the dataset.
print(df.head())
print(df.mean())
print(df.describe())

# Checking for missing values.
print(df.isna().sum())

"""
The 'description' column is simply a synopsis of the show and will not be used
in the analysis. Thus, we won't remove N/A rows for the description, as they 
might contain valuable information. We will remove N/A values for columns that
will affect the analysis, such as; number of watchers, rating, tags, etc.
Most of the missing values belong to shows that did not air yet, so these will
be irrelevant to the dataset.
"""

# Removing missing values from relevant columns.
df1 = df[['title', 'mediaType', 'eps', 'studios', 'tags', 'contentWarn',
          'watched', 'watching', 'wantWatch', 'dropped', 'rating']].dropna()

# Summary statistics for the new data.
print(df1.mean())
print(df1.describe())

# Finding top 10s in different classifications.
pd.set_option('display.max_colwidth', -1)

print(df1.nlargest(20, 'watched')['title']) # Most watched anime.
print(df1.nlargest(20, 'watched')['studios']) # Most watched anime's studios.
print(df1.nlargest(20, 'watched')['tags']) # Most watched genres.
print(df1.nlargest(20, 'watched')['contentWarn']) # Most common CW.
print(df1.nlargest(20, 'rating')['title']) # Highest rated anime.
print(df1.nlargest(20, 'rating')['studios']) # Highest rated anime's studios.
print(df1.nlargest(20, 'rating')['tags']) # Highest rated genres.
print(df1.nlargest(20, 'rating')['contentWarn']) # Highest rated CW.

# Identifying the most popular tags and highest rated tags.
popular_tags = df1.nlargest(20, 'watched')['tags']

f1 = CountVectorizer(stop_words={'a', 'if', 'of', 'on','based',
                                'fi', 'characters',
                                'main'}).build_analyzer()(str(popular_tags))
count1 = Counter(f1).most_common(10)
print(count1)

rated_tags = df1.nlargest(20, 'rating')['tags']
f2 = CountVectorizer(stop_words={'a', 'if', 'of', 'on','based',
                                'fi', 'characters',
                                'main'}).build_analyzer()(str(rated_tags))
count2 = Counter(f2).most_common(10)
print(count2)

con_warn = df1.nlargest(20, 'watched')['contentWarn']
f3 = CountVectorizer(stop_words={'abuse',
                                 'themes'}).build_analyzer()(str(con_warn))
count3 = Counter(f3).most_common(10)
print(count3)

# ----------------------------------------------------------------------------
# This section will cluster rating vs. number of watches, ie popularity.
# Then we examine the graph.
plt.scatter(df1['rating'], df1['watched'])
plt.xlabel('rating')
plt.ylabel('watched')

# Elbow plot to find out how many clusters.
sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(df1[['rating','watched']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)

# We standardise the values for an accurate scale.
scaler = MinMaxScaler()

scaler.fit(df1[['watched']])
df1['watched'] = scaler.transform(df1[['watched']])

scaler.fit(df1[['wantWatch']])
df1['wantWatch'] = scaler.transform(df1[['wantWatch']])

scaler.fit(df1[['rating']])
df1['rating'] = scaler.transform(df1[['rating']])
df1.head()

# Now we use k-mean clustering with k value 3.
km = KMeans(n_clusters = 3)
y_predicted = km.fit_predict(df1[['rating','watched']])
y_predicted

df1['cluster'] = y_predicted
df1.head()

km.cluster_centers_ # Finding the cluster centers

dfc0 = df1[df1.cluster == 0]
dfc1 = df1[df1.cluster == 1]
dfc2 = df1[df1.cluster == 2]
plt.scatter(dfc0['rating'], dfc0['watched'], color = '#CB997E', s = 10, marker = "p")
plt.scatter(dfc1['rating'], dfc1['watched'], color = '#6D597A', s = 10, marker = "p")
plt.scatter(dfc2['rating'], dfc2['watched'], color = '#E56B6F', s = 10, marker = "p")

plt.scatter(km.cluster_centers_[:,0],
            km.cluster_centers_[:,1],
            color = '#1D3557',
            marker = "X")
plt.xlabel('Rating')
plt.ylabel('Viewer Count')

colors = {'Centroid': '#1D3557',
          'Cluster A': '#E56B6F',
          'Cluster B': '#6D597A',
          'Cluster C': '#CB997E'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color = colors[label]) for label
           in labels]
plt.legend(handles, labels, labelcolor = 'linecolor', loc = 'upper left',
           fontsize = "large")
plt.title('Figure 1: Poppularity vs. Rating Clusters', weight = 'bold')
plt.savefig('rate-vs-view.png', dpi = 300, bbox_inches = 'tight') 
plt.show()

# Assess what tags the clusters include.

popular_tags_c0 = dfc0.nlargest(20, 'watched')['tags']
print(popular_tags_c0)
c0 = CountVectorizer(stop_words={'a', 'if', 'of', 'on','based',
                                'fi', 'characters',
                                'main'}).build_analyzer()(str(popular_tags_c0))
countc0 = Counter(c0).most_common(10)
print(countc0)

popular_tags_c1 = dfc1.nlargest(20, 'watched')['tags']
print(popular_tags_c1)
c1 = CountVectorizer(stop_words={'a', 'if', 'of', 'on','based',
                                'fi', 'characters',
                                'main'}).build_analyzer()(str(popular_tags_c1))
countc1 = Counter(c1).most_common(10)
print(countc1)

popular_tags_c2 = dfc2.nlargest(20, 'watched')['tags']
print(popular_tags_c2)
c2 = CountVectorizer(stop_words={'a', 'if', 'of', 'on','based',
                                'fi', 'characters',
                                'main'}).build_analyzer()(str(popular_tags_c2))
countc2 = Counter(c2).most_common(10)
print(countc2)

# ----------------------------------------------------------------------------
# This section will cluster the ratings vs. 'want to watch'.
# i.e. what consumers are interested in based on the ratings.

plt.scatter(df1['rating'], df1['wantWatch'])
plt.xlabel('rating')
plt.ylabel('wantWatch')

sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(df1[['rating','wantWatch']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)

scaler = MinMaxScaler()

scaler.fit(df1[['watched']])
df1['watched'] = scaler.transform(df1[['watched']])

scaler.fit(df1[['wantWatch']])
df1['wantWatch'] = scaler.transform(df1[['wantWatch']])

scaler.fit(df1[['rating']])
df1['rating'] = scaler.transform(df1[['rating']])
df1.head()

km = KMeans(n_clusters = 3)
y_predicted = km.fit_predict(df1[['rating','wantWatch']])
y_predicted

df1['cluster'] = y_predicted
df1.head()

km.cluster_centers_

dfc0 = df1[df1.cluster == 0]
dfc1 = df1[df1.cluster == 1]
dfc2 = df1[df1.cluster == 2]
plt.scatter(dfc0['rating'], dfc0['wantWatch'], color = '#CB997E', s = 10, marker = "p")
plt.scatter(dfc1['rating'], dfc1['wantWatch'], color = '#6D597A', s = 10, marker = "p")
plt.scatter(dfc2['rating'], dfc2['wantWatch'], color = '#E56B6F', s = 10, marker = "p")

plt.scatter(km.cluster_centers_[:,0],
            km.cluster_centers_[:,1],
            color = '#1D3557',
            marker = "X")
plt.xlabel('Rating')
plt.ylabel('"Want to Watch" List Entries')

colors = {'Centroid': '#1D3557',
          'Cluster A': '#E56B6F',
          'Cluster B': '#6D597A',
          'Cluster C': '#CB997E'}       
labels = list(colors.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color = colors[label]) for label
           in labels]
plt.legend(handles, labels, labelcolor = 'linecolor', loc = 'upper left',
           fontsize = "large")

plt.title('Figure 2: Rating vs. "Want to Watch" Clusters', weight = 'bold')
plt.savefig('rate-vs-ww.png', dpi = 300, bbox_inches = 'tight') 
plt.show()

# ----------------------------------------------------------------------------

# This section clusters watched vs. want to watch.
# i.e. what consumers are interested in based on the popularity.

plt.scatter(df1['watched'], df1['wantWatch'])
plt.xlabel('watched')
plt.ylabel('wantWatch')

sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(df1[['watched','wantWatch']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)

scaler = MinMaxScaler()

scaler.fit(df1[['watched']])
df1['watched'] = scaler.transform(df1[['watched']])

scaler.fit(df1[['wantWatch']])
df1['wantWatch'] = scaler.transform(df1[['wantWatch']])

scaler.fit(df1[['rating']])
df1['rating'] = scaler.transform(df1[['rating']])
df1.head()

km = KMeans(n_clusters = 3)
y_predicted = km.fit_predict(df1[['watched','wantWatch']])
y_predicted

df1['cluster'] = y_predicted
df1.head()

dfc0 = df1[df1.cluster == 0]
dfc1 = df1[df1.cluster == 1]
dfc2 = df1[df1.cluster == 2]
plt.scatter(dfc0['watched'], dfc0['wantWatch'], color = '#CB997E', s = 10, marker = "p")
plt.scatter(dfc1['watched'], dfc1['wantWatch'], color = '#6D597A', s = 10, marker = "p")
plt.scatter(dfc2['watched'], dfc2['wantWatch'], color = '#E56B6F', s = 10, marker = "p")

plt.scatter(km.cluster_centers_[:,0],
            km.cluster_centers_[:,1],
            color = '#1D3557',
            marker = "X")
plt.xlabel('Viewer count')
plt.ylabel('"Want to Watch" List Entries')

colors = {'Centroid': '#1D3557',
          'Cluster A': '#E56B6F',
          'Cluster B': '#6D597A',
          'Cluster C': '#CB997E'}       
labels = list(colors.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color = colors[label]) for label
           in labels]
plt.legend(handles, labels, labelcolor = 'linecolor', loc = 'upper left',
           fontsize = "large")

plt.title('Figure 2: Viewer Count vs. "Want to Watch" Clusters', weight = 'bold')
plt.savefig('watched-vs-ww.png', dpi = 300, bbox_inches = 'tight') 
plt.show()

popular_tags_c0 = dfc0.nlargest(12, 'wantWatch')['tags']
c0 = CountVectorizer(stop_words={'a', 'if', 'of', 'on','based',
                                'fi', 'characters',
                                'main'}).build_analyzer()(str(popular_tags_c0))
countc0 = Counter(c0).most_common(12)
print(countc0)

popular_tags_c1 = dfc1.nlargest(12, 'wantWatch')['tags']
c1 = CountVectorizer(stop_words={'a', 'if', 'of', 'on','based',
                                'fi', 'characters',
                                'main'}).build_analyzer()(str(popular_tags_c1))
countc1 = Counter(c1).most_common(12)
print(countc1)

popular_tags_c2 = dfc2.nlargest(12, 'wantWatch')['tags']
c2 = CountVectorizer(stop_words={'a', 'if', 'of', 'on','based',
                                'fi', 'characters',
                                'main'}).build_analyzer()(str(popular_tags_c2))
countc2 = Counter(c2).most_common(12)
print(countc2)


popular_contentWarn_c0 = dfc0.nlargest(12, 'wantWatch')['contentWarn']
c0 = CountVectorizer(stop_words={'a', 'if', 'of', 'on','based',
                                'fi', 'characters',
                                'main'}).build_analyzer()(str(popular_contentWarn_c0))
countc0 = Counter(c0).most_common(12)
print(countc0)

popular_contentWarn_c1 = dfc1.nlargest(20, 'wantWatch')['contentWarn']
c1 = CountVectorizer(stop_words={'a', 'if', 'of', 'on','based',
                                'fi', 'characters',
                                'main'}).build_analyzer()(str(popular_contentWarn_c1))
countc1 = Counter(c1).most_common(20)
print(countc1)

popular_contentWarn_c2 = dfc2.nlargest(12, 'wantWatch')['contentWarn']
c2 = CountVectorizer(stop_words={'a', 'if', 'of', 'on','based',
                                'fi', 'characters',
                                'main'}).build_analyzer()(str(popular_contentWarn_c2))
countc2 = Counter(c2).most_common(12)
print(countc2)


"""
Shounen vs. Comedy
From the previous sections, we found and counted the top 10
watched and rated genres. We'll select 2 at random then take 
a subset of users to see what their preferred genres are.
"""

shounen_anime = df1[df1['tags'].str.contains('Shounen')]
comedy_anime = df1[df1['tags'].str.contains('Slice of Life')]
print(shounen_anime)
print(comedy_anime)

plt.scatter(shounen_anime['rating'], shounen_anime['wantWatch'])
plt.xlabel('Shounen rating')
plt.ylabel('"Want to Watch" List Entries')

sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(shounen_anime[['rating','wantWatch']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)

scaler = MinMaxScaler()

scaler.fit(df1[['watched']])
df1['watched'] = scaler.transform(df1[['watched']])

scaler.fit(df1[['wantWatch']])
df1['wantWatch'] = scaler.transform(df1[['wantWatch']])

scaler.fit(df1[['rating']])
df1['rating'] = scaler.transform(df1[['rating']])
df1.head()

km = KMeans(n_clusters = 3)
y_predicted = km.fit_predict(shounen_anime[['rating','wantWatch']])
y_predicted

shounen_anime['cluster'] = y_predicted
shounen_anime.head()

km.cluster_centers_

dfc0 = shounen_anime[shounen_anime.cluster == 0]
dfc1 = shounen_anime[shounen_anime.cluster == 1]
dfc2 = shounen_anime[shounen_anime.cluster == 2]
plt.scatter(dfc0['rating'], dfc0['wantWatch'], color = '#CB997E', s = 10, marker = "p")
plt.scatter(dfc1['rating'], dfc1['wantWatch'], color = '#6D597A', s = 10, marker = "p")
plt.scatter(dfc2['rating'], dfc2['wantWatch'], color = '#E56B6F', s = 10, marker = "p")

plt.scatter(km.cluster_centers_[:,0],
            km.cluster_centers_[:,1],
            color = '#1D3557',
            marker = "X")
plt.xlabel('Rating')
plt.ylabel('"Want to Watch" List Entries')

colors = {'Centroid': '#1D3557',
          'Cluster A': '#6D597A',
          'Cluster B': '#CB997E',
          'Cluster C': '#E56B6F'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color = colors[label]) for label
           in labels]
plt.legend(handles, labels, labelcolor = 'linecolor', loc = 'upper left',
           fontsize = "xx-small")
plt.title('Figure 3', weight = 'bold')
plt.show()


# Comedy.

plt.scatter(comedy_anime['rating'], comedy_anime['wantWatch'])
plt.xlabel('comedy rating')
plt.ylabel('"Want to Watch" List Entries')

sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(comedy_anime[['rating','wantWatch']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)

scaler = MinMaxScaler()

scaler.fit(df1[['watched']])
df1['watched'] = scaler.transform(df1[['watched']])

scaler.fit(df1[['wantWatch']])
df1['wantWatch'] = scaler.transform(df1[['wantWatch']])

scaler.fit(df1[['rating']])
df1['rating'] = scaler.transform(df1[['rating']])
df1.head()

km = KMeans(n_clusters = 3)
y_predicted = km.fit_predict(comedy_anime[['rating','wantWatch']])
y_predicted

comedy_anime['cluster'] = y_predicted
comedy_anime.head()

km.cluster_centers_

dfc0 = comedy_anime[comedy_anime.cluster == 0]
dfc1 = comedy_anime[comedy_anime.cluster == 1]
dfc2 = comedy_anime[comedy_anime.cluster == 2]
plt.scatter(dfc0['rating'], dfc0['wantWatch'], color = '#CB997E', s = 10, marker = "p")
plt.scatter(dfc1['rating'], dfc1['wantWatch'], color = '#6D597A', s = 10, marker = "p")
plt.scatter(dfc2['rating'], dfc2['wantWatch'], color = '#E56B6F', s = 10, marker = "p")

plt.scatter(km.cluster_centers_[:,0],
            km.cluster_centers_[:,1],
            color = '#1D3557',
            marker = "X")
plt.xlabel('Rating')
plt.ylabel('"Want to Watch" List Entries')

colors = {'Centroid': '#1D3557',
          'Cluster A': '#6D597A',
          'Cluster B': '#CB997E',
          'Cluster C': '#E56B6F'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color = colors[label]) for label
           in labels]
plt.legend(handles, labels, labelcolor = 'linecolor', loc = 'upper left',
           fontsize = "xx-small")
plt.title('Figure 3', weight = 'bold')
plt.show()
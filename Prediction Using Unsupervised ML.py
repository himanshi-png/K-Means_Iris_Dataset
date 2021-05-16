#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Downloading the datset


# In[2]:


# importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets


# In[3]:


# import the dataset
iris_raw=pd.read_csv("Iris.csv")


# In[4]:


iris_raw


# In[5]:


# summary of data
iris_raw.describe()


# In[6]:


iris_raw.isnull()


# In[7]:


# Finding corelation between variables


# In[8]:


corr = iris_raw.corr()
corr


# In[9]:


sns.pairplot(iris_raw,hue='Species');


# In[10]:


# Pair plot


# In[11]:


sns.set()
sns.heatmap(iris_raw.corr(),annot= True, cmap="pink");


# In[12]:


# Find the optimum number of clusters for K-Means and then value of K


# In[14]:


# Defining "x"
x = iris_raw.iloc[:, [0, 1, 2, 3]].values

# Finding the optimum number of clusters for k-means classification
from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, 
# `allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.grid()
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show();


# In[15]:


# Creating the kmeans classifier
kmeans = KMeans(n_clusters=5, init ='k-means++', max_iter=300, n_init=10,random_state=0 )


# In[16]:


y_kmeans = kmeans.fit_predict(x)


# In[17]:


# Visualising the dataset


# In[18]:



plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa');
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour');
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica');

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids');

plt.legend();


# In[ ]:





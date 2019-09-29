#KMeans
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from statistics import mean

#data=[ [2.5,3],[6,7],[9,8],[3,4],[8.5,5],[2,5] ]
data=[ [1,1], [6,8], [8,5], [2,1], [2,2], [4,8],[5,9], [9,5] ,[9,4]]

from sklearn.cluster import KMeans        

clf=KMeans(n_clusters=3)
clf.fit(data)


for point in data:
    ppoint=np.array(point)
    ppoint.shape=(1,-1)
    label=clf.predict(ppoint)
    plt.scatter(point[0],point[1], marker="$"+str(label)+"$",s=200)

print(clf.cluster_centers_)

for center in clf.cluster_centers_:
    plt.scatter(center[0],center[1])

plt.show()

data.append([2.5,5])
clf.fit(data)

for point in data:
    ppoint=np.array(point)
    ppoint.shape=(1,-1)
    label=clf.predict(ppoint)
    plt.scatter(point[0],point[1], marker="$"+str(label)+"$",s=200)

print(clf.cluster_centers_)

for center in clf.cluster_centers_:
    plt.scatter(center[0],center[1])

plt.show()


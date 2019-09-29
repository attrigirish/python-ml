#KMeans
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from statistics import mean

#data=[ [2.5,3],[6,7],[9,8],[3,4],[8.5,5],[2,5] ]
data=[ [1,1], [6,8], [8,5], [2,1], [2,2], [4,8],[5,9], [9,5] ,[9,4]]

#for point in data:
#    plt.scatter(point[0],point[1], color='red')
#plt.show()


class KMean:
    centeroids={}
    n=0
    ite=0

    def __init__(self,n=2,ite=100):
        self.n=n
        self.ite=ite

    def fit(self,data):

        classifications={}

        for index in range(self.n):
            self.centeroids[index] = data[index]
            classifications[index] = []


        for i in range(1,self.ite+1):
            for point in data:
                distances=[]
                for centeroid in self.centeroids:
                    cpoint=self.centeroids[centeroid]
                    distance=sqrt ( (point[0]-cpoint[0])**2 + (point[1]-cpoint[1])**2 )
                    distances.append(distance)
                label=distances.index(min(distances))
                classifications[label].append(point)


            old_centeroids=self.centeroids.copy()

            for key in classifications:
                self.centeroids[key] = np.average(classifications[key], axis=0)

            isoptimized=True
            for key in self.centeroids:
                if(old_centeroids[key][0]!=self.centeroids[key][0] or old_centeroids[key][1]!=self.centeroids[key][1]):
                    isoptimized=False

            print("Iteration : ", i)
            

            if(isoptimized==True):
                break

    def predict(self,prediction):
                distances=[]
                for centeroid in self.centeroids:
                    cpoint=self.centeroids[centeroid]
                    distance=sqrt ( (prediction[0]-cpoint[0])**2 + (prediction[1]-cpoint[1])**2 )
                    distances.append(distance)
                label=distances.index(min(distances))
                return label
        

clf=KMean(n=3)
clf.fit(data)



for point in data:
    label=clf.predict(point)
    plt.scatter(point[0],point[1], marker="$"+str(label)+"$")

for key in clf.centeroids:
    plt.scatter(clf.centeroids[key][0],clf.centeroids[key][1])

plt.show()

data.append([2.5,5])
clf.fit(data)

for point in data:
    label=clf.predict(point)
    plt.scatter(point[0],point[1], marker="$"+str(label)+"$")

for key in clf.centeroids:
    plt.scatter(clf.centeroids[key][0],clf.centeroids[key][1])

plt.show()

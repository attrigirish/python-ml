import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import sqrt


features=pd.DataFrame([
            [2.9, 2.2],
              [2.6,2.4],
              [2.3, 2.3],
              [2.5, 2.6],
              [2.1, 2.3 ],              
              [2.4, 2],
              [2.6,2],
              [2.6,2.2],
              [2,2.2],
             [2.5,2.0],
             [2.25,2.25]
        ], columns=['x','y'])

plt.scatter(features['x'],features['y'])
#plt.show()



#Algorithm
class KMeansClassifier:
    c=0                                             #No of Clusters
    iter=0                                     #Maximum Iterations
    centeroids={}                             #Model Centers
    labels=[]                                    #Model Labels
    features={}
    margin=0

    def __init__(self, c=2, iter=100,margin=0.1):
        self.c=c
        self.iter=iter
        self.margin=margin
        for index in range(1,c+1):
            label='class'+str(index)
            self.labels.append(label)
            self.centeroids[label]=[0,0]
            self.features[label]=[]
        

    def fit(self,features):
        if(len(features)<self.c):
            print("Less Features")
            return

        for index in range(self.c):
            label=self.labels[index]
            self.centeroids[label]=[features.loc[index]['x'], features.loc[index]['y']]

        for count in range(self.iter):
            self.features={}

            #Calculate Distances
            for index in range(len(features)):
                distances=[]
                for key in self.centeroids:
                    feature=[features.loc[index]['x'], features.loc[index]['y']]
                    center=self.centeroids[key]
                    distance=sqrt( (feature[0]-center[0])**2 + (feature[1]-center[1])**2 )
                    distances.append(distance)
                nearest_center = distances.index(min(distances))
                nearest_class = self.labels[nearest_center]
                if(nearest_class in self.features.keys()):
                    self.features[nearest_class].append(feature)
                else:
                    self.features[nearest_class]=[feature]

            old_centeroids=self.centeroids.copy()

            #Recomputer Centers
            for key in self.features:
                data=self.features[key]
                x=0
                y=0
                for point in data:
                    x+=point[0]
                    y+=point[1]
                self.centeroids[key] =  [  x/len(data), y/len(data)  ]                     
                #self.centeroids[key] = [ data[:, 0].mean(), data[:, 1].mean() ]                

            isoptimized=True
            for key in self.centeroids:
                print(old_centeroids[key], self.centeroids[key])
                print("X - Margin : ", self.centeroids[key][0]-old_centeroids[key][0])
                print("Y - Margin : ", self.centeroids[key][1]-old_centeroids[key][1])
                if(abs(self.centeroids[key][0]-old_centeroids[key][0])>self.margin and abs(self.centeroids[key][1]-old_centeroids[key][1])>self.margin):
                    isoptimized=False

            print("Iteration : ", count)
            if(isoptimized==True):
                break

#Model
model=KMeansClassifier(c=3,iter=50,margin=0.001)
model.fit(features)

print(model.labels)
print(model.centeroids)
print(model.features)


colors=['r','b','g','y','c','p','k']
color_map={}
count=0
for label in model.features:
    color_map[label]=colors[count]
    count+=1


for key in model.features:    
    for feature in model.features[key]:
        plt.scatter(feature[0], feature[1], color=color_map[key])

for key in model.centeroids:
    plt.scatter(model.centeroids[key][0], model.centeroids[key][1], 200, color=color_map[key])
    
plt.show()

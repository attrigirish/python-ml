from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt


class KNN:
    k=0
    features=0
    labels=0
    length=0

    def __init__(self, k=3):
        self.k=k

    def fit(self,features, labels):
        if(len(features)!=len(labels)):
            raise Exception("Feature and Label lenghts should be identical")
        self.features=features
        self.labels=labels
        self.length=len(self.features)

    def predict(self,pred_features):
        pred_labels=[]
        for oindex in range(len(pred_features)):
            distances=[]
        
            for index in range(0,self.length):
                label=self.labels[index]
                pointA = self.features.loc[index]
                pointB = pred_features.loc[oindex]
                distance=sqrt((pointA[0]-pointB[0])**2+(pointA[1]-pointB[1])**2)
                distances.append([distance,label])

            #Arrange distances in increasing order
            distances=sorted(distances,key=lambda x:x[0])

            label_count={}
            for label in set(self.labels):
                label_count[label]=0


            #Check the labels for first k elements
            for distance in distances[:self.k]:
                label=distance[1]
                label_count[label]=label_count[label]+1


            #Find the label/class with maximum votes
            label_count=label_count.items()
            label_count=sorted(label_count, key=lambda x:x[1], reverse=True)
            pred_labels.append(label_count[0][0])            
            
        return pred_labels




data=pd.DataFrame(
    [
        [2,2,'A'],
        [3,5,'A'],
        [2,4,'A'],
        [4,1,'A'],
        [6,5,'B'],
        [5,7,'B'],
        [5,5,'B'],
        [8,4,'B'],
        [1,9,'C'],
        [2,6,'C'],
        [3,8,'C'],
        [1.5,7,'C']
        ],
    columns=['X','Y','Class']
    )


#Prediction Data
pred_features=pd.DataFrame([[3.6,6],[6,7]])


#Visualizing Source Data
unique_labels=list(set(data['Class']))

colors=['red','green','blue','black','yellow','orange']
label_colors={}
for index in range(len(unique_labels)):
    label_colors[unique_labels[index]]=colors[index]



for index in range(len(data)):
    label=data.loc[index]['Class']
    x=data.loc[index]['X']
    y=data.loc[index]['Y']
    plt.scatter(x,y,color=label_colors[label], marker='$'+label+'$')

for index in range(len(pred_features)):
    plt.scatter(pred_features.loc[index][0], pred_features.loc[index][1], color='cyan', marker="$?$")    

plt.show()



#Training Logic

#Feature and Label Extractions
features=data.drop(['Class'], axis=1)
labels=data['Class']

#Training the Model
model = KNN(k=3)
model.fit(features,labels)

#Predicting Data
pred_labels=model.predict(pred_features)
print("Prediction Result : ", pred_labels)


for index in range(len(data)):
    label=data.loc[index]['Class']
    x=data.loc[index]['X']
    y=data.loc[index]['Y']
    plt.scatter(x,y,color=label_colors[label], marker='$'+label+'$')

for index in range(len(pred_features)):    
    plt.scatter(pred_features.loc[index][0], pred_features.loc[index][1], color=label_colors[pred_labels[index]], marker="$"+pred_labels[index]+"$")    
plt.show()



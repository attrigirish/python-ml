#Size Calculator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz
import graphviz


data=pd.read_csv("https://raw.githubusercontent.com/attrigirish/python-ml/master/resources/Size%20Data.csv")

#Convert Categories to Numbers
brand_encoder=LabelEncoder()
brand_encoder.fit(data['Brand'])
data['Brand']=brand_encoder.transform(data['Brand'])

#Split the Data in Features and Labels
features=data[ ['Height','Weight','Brand'] ]
labels=data['Fit']

#Split the Features and Labels in Training and Test sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)


#Classifier - Using Gini Impurity
model=DecisionTreeClassifier(criterion='gini')
model.fit(train_features,train_labels)

#Accuracy
accuracy=model.score(test_features,test_labels)
print("Accuracy : ",accuracy*100)

#GraphViz
dot_data = export_graphviz(model,
                                feature_names=['Height','Weight','Brand'],
                                class_names=['S','M','L'],
                                rounded=True,
                                filled=True)

graph = graphviz.Source(dot_data,filename='gini',format='jpg')
graph.render(directory=r'C:\Users\girish\Desktop')




#Classifier - Using entropy impurity
model=DecisionTreeClassifier(criterion='entropy')
model.fit(train_features,train_labels)

#Accuracy
accuracy=model.score(test_features,test_labels)
print("Accuracy : ",accuracy*100)

#GraphViz
dot_data = export_graphviz(model,
                                feature_names=['Height','Weight','Brand'],
                                class_names=['S','M','L'],
                                rounded=True,
                                filled=True)

graph = graphviz.Source(dot_data,filename='entropy',format='jpg')
graph.render(directory=r'C:\Users\girish\Desktop')


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


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


while(True):
    print("1. Linear Kernel")
    print("2. Polynomial Kernel")
    print("3. RBF Kernel")
    print("4. Sigmoid Kernel")

    print()
    choice=int(input("Enter Choice : "))
    print()

    if(choice==1):
        model=SVC(gamma='auto',kernel='linear')
    elif(choice==2):
        model=SVC(gamma='auto',kernel='poly')
    elif(choice==3):
        model=SVC(gamma='auto',kernel='rbf')
    elif(choice==4):
        model=SVC(gamma='auto',kernel='sigmoid')
    elif(choice==0):
        break

    print(model)
    model.fit(train_features,train_labels)

    #Accuracy
    accuracy=model.score(test_features,test_labels)
    print("Accuracy : ",accuracy*100)
        
    print()
    print()

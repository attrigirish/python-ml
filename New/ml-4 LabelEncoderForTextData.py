import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

x = np.array(['A','B','C','D','A','C','B','D','C','E'])
y = np.array([1.1,3.0,4.2,7,1.2,4.3,3.1,7.7,4.5,10.2])


#Converting Text Data into Numbers
le=LabelEncoder()
le.fit(x)           #Mapping


#Create Classifier Object
clf = LinearRegression()

#Train the Classifier
x=le.transform(x)
x.shape=(-1,1)
clf.fit(x,y)

#Evaluating the Accuracy of the Model
score=clf.score(x,y)
print("Accuracy of The Model : ", score)

#Predictions
predict_x=np.array(['A','E'])
predict_x=le.transform(predict_x)
predict_x.shape=(-1,1)
predict_y=clf.predict(predict_x)
print("Prediction Data : ", predict_y)

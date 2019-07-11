import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([1,2,3,4,5])
x.shape=(-1,1)
y = np.array([3,4,5,6,7])

#Create Classifier Object
clf = LinearRegression()

#Train the Classifier
clf.fit(x,y)

#Evaluating the Accuracy of the Model
score=clf.score(x,y)
print("Accuracy of The Model : ", score)

#Predictions
predict_x=np.array([6,8,9])
predict_x.shape=(-1,1)
predict_y=clf.predict(predict_x)
print("Prediction Data : ", predict_y)

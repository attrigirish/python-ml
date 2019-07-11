import numpy as np
from sklearn.linear_model import LinearRegression

#Converting Text Data to Numbers
def Mapper(data):
    unique=list(set(data))
    unique.sort()
    count=1
    map_data={}
    for value in unique:
        map_data[value]=count
        count+=1
    return map_data

def ConvertTextToNumber(data,maps):
    for i in range(len(data)):
        data[i]=maps[data[i]]



xdata=['A','B','C','D','A','C','B','D','C','E']
ydata=[1.1,3.0,4.2,7,1.2,4.3,3.1,7.7,4.5,10.2]

maps=Mapper(xdata)
ConvertTextToNumber(xdata,maps)

x = np.array(xdata)
x.shape=(-1,1)
y = np.array(ydata)



#Create Classifier Object
clf = LinearRegression()

#Train the Classifier
clf.fit(x,y)

#Evaluating the Accuracy of the Model
score=clf.score(x,y)
print("Accuracy of The Model : ", score)

#Predictions
predictdata=['A','E']
ConvertTextToNumber(predictdata,maps)
predict_x=np.array(predictdata)
predict_x.shape=(-1,1)
predict_y=clf.predict(predict_x)
print("Prediction Data : ", predict_y)

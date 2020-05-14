from sklearn import datasets
from sklearn.svm import SVC
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

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


model=SVC(kernel='rbf', C=1, gamma=0.1)
pipeline = Pipeline([('task',model)]) 

params = {'task_C':(0.1, 0.5, 1, 2, 5, 10, 20), 
          'task_gamma':(0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1)} 


gridModel = GridSearchCV(pipeline, params) 

gridModel.fit(train_features, train_labels) 

print("Best Score : ", gridModel.best_score_)

best = gridModel.best_estimator_.get_params()
print(best)

for k in sorted(params.keys()): 
    print(k, best[k])



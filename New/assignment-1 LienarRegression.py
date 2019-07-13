import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression


df=pd.DataFrame(data=[
        ['Rahul','X',53],
        ['Sunil','X',73],
        ['Rahul','XI',57],
        ['Sunil','XI',64],
        ['Rahul','XII',62],
        ['Sunil','XII',59],
    ],columns=['Name','Class','Marks'])

#print(df)


#Mappings
le_name=LabelEncoder()
le_class=LabelEncoder()
le_name.fit(df['Name'])
le_class.fit(df['Class'])

#Conversion
df['Name']=le_name.transform(df['Name'])
df['Class']=le_class.transform(df['Class'])

#print(df)

#Extracting Features and Labels
X=df.drop(['Marks'], axis=1)
#print(X)
y=df['Marks']
#print(y)


#Model
clf=LinearRegression()
clf.fit(X,y)

#Accuracy
score=clf.score(X,y)
print(score)

#Prediction
predict_data=X.tail(2)
print(predict_data)
predict_output=clf.predict(predict_data)
print(predict_output)







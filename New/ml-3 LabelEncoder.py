from sklearn.preprocessing import LabelEncoder

data=['A','E','A','C','B','D','B','C','C','A','A','E','F']
le = LabelEncoder()
le.fit(data)

#Check Classes
print(le.classes_)

#Transform Data
data=le.transform(data)
print(data)

#Inverse Transform
data=le.inverse_transform(data)
print(data)

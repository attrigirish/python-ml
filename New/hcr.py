#Handwritten Character Recognition.

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split





digits = datasets.load_digits()

#print(digits)

index=100

#Data
print(digits['target_names'])
print(digits['data'][index])
print(digits['target'][index])
print(digits['images'][index])



plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[index], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


#Data Extraction
digits_data = digits.data
print(digits_data.shape)                        #1797X64

digits_target = digits.target
print(digits_target.shape)                      #1797

number_digits = len(digits['target_names'])

digits_images = digits.images
print(digits_images.shape)                      #1797X8X8


#Data Visualize
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    ax.text(0, 7, str(digits.target[i]))
plt.show()


#Dimensionality Reduction : Reducing the no. of columns to 2
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(digits.data)
print(reduced_data.shape)
print(reduced_data[index])


#Scaling 
reduced_data = scale(reduced_data)
print(reduced_data[index])


#Split Data
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(reduced_data, digits_target, digits_images)

#Model Training
model=KNeighborsClassifier()
model.fit(X_train,y_train)

#Model Accuracy
score=model.score(X_test,y_test)
print("Model Accuracy : ", score)


#Prediction
indexes=[50,150,250,350,450,550,650]
pred_data=reduced_data[indexes]
result=model.predict(pred_data)
print("Predicted Numbers")
print(result)
print("Actual Numbers")
print(digits_target[indexes])

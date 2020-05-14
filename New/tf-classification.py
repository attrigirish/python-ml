import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Exploring Data

print("Train Features Dimensions : ", train_images.ndim)
print("Train Features Shape : ", train_images.shape)
print("Train Features Size : " , len(train_images))

print("\n\n")

print("Test Features Dimensions : ", test_images.ndim)
print("Test Features Shape : ", test_images.shape)
print("Test Features Size : " , len(test_images))

#Normalize/Scale
train_images = train_images / 255.0
test_images = test_images / 255.0


#Step 1 : Create the Neural Network

model = keras.Sequential([
    keras.layers.Dense(1, activation=tf.nn.relu),      			#Input
    keras.layers.Dense(10, activation=tf.nn.relu), 				#Hidden
    keras.layers.Dense(1, activation=tf.nn.softmax) 			#Output
])

#Step 2 : Compile the Model

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#Step 3 : Training the Model

model.fit(train_images, train_labels, epochs=100)

test_loss, test_acc = model.evaluate(test_images, test_labels)

#Step 4 : Predictions

predictions = model.predict(test_images)
p_index=500
print("Actual Class for The First Test Image : ", test_labels[p_index],class_names[test_labels[p_index]])
print("Confidence Score : ",predictions[p_index])
print("Model Output : ", np.argmax(predictions[p_index]), class_names[np.argmax(predictions[p_index])])
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import layers
import random
from sklearn.model_selection import train_test_split

originalFile = "balancedsixClasses.csv"

originalData = np.genfromtxt(originalFile, dtype=np.float64, delimiter=",", skip_header=1)
trainingData, testData = train_test_split(originalData, test_size=0.30, random_state=42)


# shuffle the order of data
random.shuffle(trainingData)

testLabels = testData[:,-1]
testFeatures = testData[:,0:-1]
labels = trainingData[:,-1]
features = trainingData[:,0:-1]



#create neural network model using Keras
model = tf.keras.Sequential()
#add input layer
model.add(layers.Dense(79, input_shape=(79,), activation='tanh'))
#add hidden layers
model.add(layers.Dense(50, activation='tanh'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(50, activation='tanh'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(50, activation='tanh'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(50, activation='tanh'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(50, activation='tanh'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(50, activation='tanh'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(50, activation='tanh'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(50, activation='tanh'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(50, activation='tanh'))
model.add(layers.Dropout(0.2))
#add output layer
model.add(layers.Dense(6, activation='softmax'))

# # #configure model learning process with adam optimizer, categorical crossentropy loss function, and accuracy as a metric
model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model = Sequential()
# model.add(Dense(79, input_shape = features.shape[1:]))

# model.add(Dense(50))
# model.add(Dense(50))
# model.add(Dense(50))

# model.add(Dense(50))
# model.add(Dense(50))

# model.add(Dense(15))
# model.add(Activation('sigmoid'))

# model.compile(loss="sparse_categorical_crossentropy",
#             optimizer="adam",
#             metrics=['accuracy'])

model.fit(features, labels, batch_size=32, epochs=10, validation_split=0.1)

predictions = model.predict_classes(testFeatures,batch_size=32)
confusionMatrix = tf.confusion_matrix(testLabels,predictions,num_classes=6,dtype=tf.int32)
print("Results as confusion matrix")

with tf.Session():
   print('Confusion Matrix: \n\n', tf.Tensor.eval(confusionMatrix,feed_dict=None, session=None))
print(confusionMatrix)

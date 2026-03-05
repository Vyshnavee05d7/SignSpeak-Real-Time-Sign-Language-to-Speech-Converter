import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from attention import attention #loading attention layer
from keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, RepeatVector
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from keras.callbacks import ModelCheckpoint

labels = ['I', 'apple', 'can', 'get', 'good', 'have', 'help', 'how', 'like', 'love', 'my', 'no', 'sorry', 'thank-you', 'want', 'yes', 'you', 'your']

'''
X = []
Y = []

for i in range(0, 18):
    features = np.load("model/X"+str(i)+".npy")
    label = np.load("model/Y"+str(i)+".npy")
    print(str(features.shape)+" "+str(label.shape))
    if len(X) == 0:
        X = features
        Y = label
    else:
        X = np.concatenate((X, features), axis=0)
        Y = np.concatenate((Y, label), axis=0)    

np.save("X", X)
np.save("Y", Y)
print(X.shape)
print(Y.shape)
'''

X = np.load("X.npy")
Y = np.load("Y.npy")

sc = StandardScaler()
X = sc.fit_transform(X)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

Y = to_categorical(Y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test

#training encoder from sign to text
encoder_model = Sequential()
#defining cnn layer
encoder_model.add(Convolution2D(64, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
encoder_model.add(MaxPooling2D(pool_size = (1, 1)))
encoder_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
encoder_model.add(MaxPooling2D(pool_size = (1, 1)))
encoder_model.add(Flatten())
encoder_model.add(RepeatVector(3))
encoder_model.add(attention(return_sequences=True,name='attention')) # ========define transformer Attention layer
encoder_model.add(LSTM(32))
encoder_model.add(Dense(units = 64, activation = 'relu'))
encoder_model.add(Dense(units = y_train.shape[1], activation='softmax'))
#compiling, training and loading model
encoder_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/encoder_weights.keras") == False:
    model_check_point = ModelCheckpoint(filepath='model/encoder_weights.keras', verbose = 1, save_best_only = True)
    hist = encoder_model.fit(X_train, y_train, batch_size = 32, epochs = 150, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/encoder_hist.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
else:
    encoder_model.load_weights("model/encoder_weights.keras")
#perform prediction on test data
predict = encoder_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)
































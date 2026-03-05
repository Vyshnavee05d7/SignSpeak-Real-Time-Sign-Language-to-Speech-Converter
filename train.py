import pandas as pd
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from nltk.stem import PorterStemmer
import os
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn.preprocessing import StandardScaler
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Conv2D, UpSampling2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import cv2

#define object to remove stop words and other text processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

#define function to clean text by removing stop words and other special symbols
def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens
'''
path = "Dataset/images"
X = []
Y = []


for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j],0)
            img = cv2.resize(img, (32, 32))
            name = directory[j].replace(".jpg", ".txt")
            if os.path.exists("Dataset/text/"+name):
                with open("Dataset/text/"+name, "rb") as file:
                    data = file.read()
                file.close()
                data = data.decode()
                data = data.strip("\n").strip().lower()
                data = cleanText(data)
                X.append(data)
                Y.append(img)
                print(str(len(data))+" "+str(j))
            else:
                print("============================================================ "+name)

X = np.asarray(X)
Y = np.asarray(Y)
np.save("model/X", X)
np.save("model/Y", Y)
'''
X = np.load("model/X.npy")
Y = np.load("model/Y.npy")

vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
X = vectorizer.fit_transform(X).toarray()

XX = []
for i in range(len(X)):
    img = X[i]
    img = np.reshape(img, (13, 3))
    img = cv2.resize(img, (32, 32))
    XX.append(img)
X = np.asarray(XX)    
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
Y = np.reshape(Y, (Y.shape[0], Y.shape[1], Y.shape[2], 1))
X = X.astype('float32')
X = X/255
Y = Y.astype('float32')
Y = Y/255


print(X.shape)
print(Y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)#splitting dataset into train test
print("80% images using for training : "+str(X_train.shape[0]))
print("20% images using for testing : "+str(X_test.shape[0]))


input_img = Input(shape=(32, 32, 1))
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
cnn_model = Model(input_img, decoded)
cnn_model.compile(optimizer='adam', loss='binary_crossentropy')
print(cnn_model.summary())
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 380, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model.load_weights("model/cnn_weights.hdf5")

test_data = 'woman has a heart face with long hair.  she has a pair of big normal eyes,with dense thin and arched eyebrows. her mouth is thick and narrow, with a medium normal nose and her ears are small. she hasnt glasses and hasnt beard.'
test_data = test_data.lower().strip()
test_data = cleanText(test_data)
test_data = vectorizer.transform([test_data]).toarray()
test_data = np.reshape(test_data, (13, 3))
test_data = cv2.resize(test_data, (32, 32))
temp = []
temp.append(test_data)
temp = np.asarray(temp)    
temp = np.reshape(temp, (temp.shape[0], temp.shape[1], temp.shape[2], 1))
temp = temp.astype('float32')
temp = temp/255
print(temp.shape)
predict = cnn_model.predict(temp)
print(predict.shape)
predict = predict[0]

predict = cv2.resize(predict, (300, 300))
cv2.imshow("aa", predict)
cv2.waitKey(0)











            

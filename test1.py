import os
import cv2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn.preprocessing import StandardScaler
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Conv2D, UpSampling2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

sentences = ['I use to speak Russian but I cant read it very well',
          'an apple a day keep the doctor away',
          'how can help you',
          'get me glass of grape juice',
          'good education may provide you good opportunities',
          'have glass of milk every morning and evening',
          'each day help someone to earn rewards',
          'how to score best for best lifetime achievements',
          'live like a better person',
          'love everyone in your life and dont hurt anyone',
          'my brother works very hard to get better earnings',
          'no to everything is bad habit say yes always',
          'say sorry for mistakes you did to others in your good times',
          'say thank-you to all who pray for your success',
          'you want best result then keep working hards without thinking about failure',
          'yes you need to help others which indicates your best behaviour',
          'you have vegetable more compare to non veg foods in your daily diet',
          'your chance of getting jobs high compare to other candidates']

labels = ['I', 'apple', 'can', 'get', 'good', 'have', 'help', 'how', 'like', 'love', 'my', 'no', 'sorry', 'thank-you', 'want', 'yes', 'you', 'your']
'''

words = []
signs = []
sign_label = []
old = "old"

for root, dirs, directory in os.walk("Dataset"):
    for j in range(len(directory)):
        if 'Thumbs.db' not in directory[j]:
            current = os.path.basename(root)
            if current != old:
                img = cv2.imread(root+"/"+directory[j],0)
                img = cv2.resize(img, (128, 128))
                sign_label.append(img)
                old = current
            name = int(os.path.basename(root))
            data = sentences[name]
            data = data.strip().lower()
            words.append(data)
            signs.append(img)
            print(data+" "+str(name))

words = np.asarray(words)
signs = np.asarray(signs)

np.save("model/words", words)
np.save("model/signs", signs)
'''
words = np.load("model/words.npy")
signs = np.load("model/signs.npy")
sign_label = np.load("model/sign_label.npy")
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=39)
words = vectorizer.fit_transform(words).toarray()
print(words)
print(words.shape)

XX = []
for i in range(len(words)):
    word = words[i]
    word = np.reshape(word, (13, 3))
    word = cv2.resize(word, (128, 128))
    XX.append(word)
words = np.asarray(XX)    
words = np.reshape(words, (words.shape[0], words.shape[1], words.shape[2], 1))
signs = np.reshape(signs, (signs.shape[0], signs.shape[1], signs.shape[2], 1))
words = words.astype('float32')
words = words/255
signs = signs.astype('float32')
signs = signs/255

X_train, X_test, y_train, y_test = train_test_split(words, signs, test_size = 0.2, random_state = 1)#splitting dataset into train test
print("80% images using for training : "+str(X_train.shape[0]))
print("20% images using for testing : "+str(X_test.shape[0]))


input_img = Input(shape=(128, 128, 1))
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
if os.path.exists("model/decoder_weights.keras") == False:
    model_check_point = ModelCheckpoint(filepath='model/decoder_weights.keras', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 380, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/decoder_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model.load_weights("model/decoder_weights.keras")

for i in range(len(labels)):
    test_data = labels[i]
    test_data = test_data.lower().strip()
    test_data = vectorizer.transform([test_data]).toarray()
    test_data = np.reshape(test_data, (13, 3))
    test_data = cv2.resize(test_data, (128, 128))
    temp = []
    temp.append(test_data)
    temp = np.asarray(temp)    
    temp = np.reshape(temp, (temp.shape[0], temp.shape[1], temp.shape[2], 1))
    temp = temp.astype('float32')
    temp = temp/255
    print(temp.shape)
    predict = cnn_model.predict(temp)
    #predict = sign_label[i]
    print(predict.shape)
    predict = predict[0]
    #predict = cv2.resize(predict, (200, 200))
    cv2.imshow("aa", predict)
    cv2.waitKey(0)
    #plt.imshow(predict, cmap="gray")
    #plt.show()
    












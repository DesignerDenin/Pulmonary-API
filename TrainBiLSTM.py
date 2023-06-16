import random
import os
import numpy as np
from pyrsistent import plist
import librosa.display
import soundfile
from sklearn.model_selection import train_test_split
import librosa
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from keras.utils import np_utils
from keras.layers import GRU, LSTM, Embedding,Bidirectional 

featurelist=pickle.load( open( "features/featlist2.pkl", "rb" ) )
labellist=pickle.load( open( "labels/labellist2.pkl", "rb" ) )

from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU  

lb = LabelEncoder()
ydata = np_utils.to_categorical(lb.fit_transform(labellist))
print(ydata)
xdata=np.array(featurelist)
# print(xdata)
X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.1, random_state=42)

X_train = np.reshape(X_train, (len(X_train), len(X_train[0]), 1))
print(X_train.shape)
X_val = np.reshape(X_test, (len(X_test), len(X_test[0]), 1))

print(X_val[0].shape)

print(X_train.shape)
num_labels = y_train.shape[1]
print(num_labels)

# from keras.metrics import categorical_accuracy
# from keras.layers import LSTM, Input, Bidirectional
# from tensorflow.keras.optimizers import Adam
# from keras.callbacks import EarlyStopping,ModelCheckpoint
# from keras.layers import Dense, Activation, Dropout
# learning_rate = 0.001
# model = Sequential()
# model.add(Bidirectional(LSTM(256, activation="relu"),input_shape=(40, 1)))
# model.add(Dropout(0.6))
# model.add(Dense(4))
# model.add(Activation('softmax'))

# optimizer = Adam(lr=learning_rate)
# callbacks=[EarlyStopping(patience=2, monitor='val_loss')]
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
# print("model built!")

# model.fit(X_train, y_train, batch_size=16, epochs=200, validation_data=(X_val, y_test))

# # serialize model to JSON
# model_json = model.to_json()
# with open("BiLSTMmodel.json", "w") as json_file:
#        json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("BiLSTMmodel.h5")
# print("Saved model to disk")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM,Bidirectional

import json
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
lstm_sze = 70

Bot_lstmmodel = Sequential()
Bot_lstmmodel.add(Convolution1D(64, 3, padding="same",activation="relu",input_shape=(40, 1)))
Bot_lstmmodel.add(MaxPooling1D(pool_size=(2)))
Bot_lstmmodel.add(Bidirectional((LSTM(lstm_sze))))
Bot_lstmmodel.add(Dropout(0.1))
Bot_lstmmodel.add(Dense(4, activation="sigmoid"))

Bot_lstmmodel.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])


Bot_lstmmodel.fit(X_train, y_train, batch_size=16, epochs=200, validation_data=(X_val, y_test))

model_json = Bot_lstmmodel.to_json()
with open("BiLSTM.json", "w") as json_file:
       json_file.write(model_json)

Bot_lstmmodel.save_weights("BiLSTM.h5")
print("Saved model to disk")


from tkinter import *
from tensorflow.keras.models import model_from_json
import librosa
import numpy as np

def predict(val):
    json_model_file = open('BiLSTM.json', 'r')
    loaded_model_json = json_model_file.read()
    json_model_file.close()
    modelLSTM = model_from_json(loaded_model_json)

    # load weights into new model
    modelLSTM.load_weights("BiLSTM.h5")
    print("====================================================")
    print("Loading Model:", "SUCCESS")
    print("File Name: ", val )

    data_x, sampling_rate = librosa.load(val,res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=40).T,axis=0)
    
    feat=np.array(mfccs)
    feat=np.reshape(feat,(1,40,1))
    dlist=['COPD','Healthy','Pneumonia','URTI']

    ypred=modelLSTM.predict(feat)
    print("Prediction: ",ypred)
    res=np.argmax(ypred,axis=1)[0]
    print("Type: ",res)
    result=dlist[res]

    print("Audio Upload:", "SUCCESS")
    print("Loading Bidirectional LSTM Model:", "SUCCESS")
    print("MFCC Feature extraction:", "SUCCESS" )
    print("Prediction using BiLSTM model:", "SUCCESS")
    print("Result:", result.upper())
    print("====================================================")

    return result
from keras.models import load_model
import numpy as np

# Simple testing for a single numpy array already processed by preprocess.py

ROOT = "D:/BIRDTEST/"
data = np.load(ROOT + '/Preproc/skjare/376568.wav.npy')

print(data.shape)

model = load_model(ROOT + '/Model/my_model.h5')
pred = model.predict_classes(data, verbose=1)

print("File is this class: ", pred)

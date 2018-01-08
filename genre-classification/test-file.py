
from __future__ import print_function
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

# And the display module for visualization
import librosa.display
import audioread
import tkinter
from tkinter.filedialog import askopenfilename

genres = ['metal', 'disco', 'classical', 'hiphop', 'jazz','country', 'pop', 'blues', 'reggae', 'rock']

root = tkinter.Tk()
root.withdraw()
audio_path = askopenfilename()

# audio_path = 'gtzan/blues/blues.00000.au'

# audioread.audio_open(audio_path)

# from subprocess import call
# r=call('ffmpeg -i "test.mp3" -acodec pcm_u8 -ar 22050 "test.wav"',shell=True)

signal, sr = librosa.load(audio_path)
melspec = librosa.feature.melspectrogram(signal[:660000], sr=sr, n_fft=2048, hop_length=512).T[:128, ]

models = ['gtzan_exec_5_epochs_100.h5', 'gtzan_exec_10_epochs_200.h5']

for x in models:
    model = load_model(x)

    X_train = []
    X_train.append(melspec)
    X_train = np.array(X_train)
    predictions = model.predict(X_train)
    y_classes = predictions.argmax(axis=-1)
    print(y_classes, genres[y_classes[0]])

    plt.figure()
    plt.title('Predictions for: '+x)
    plt.ylabel('Prediction')
    plt.xlabel('Genres')
    plt.bar(genres, predictions[0])
plt.show()

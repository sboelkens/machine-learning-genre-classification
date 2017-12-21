
from __future__ import print_function
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

# And the display module for visualization
import librosa.display
import tkinter
from tkinter.filedialog import askopenfilename

genres = ['metal', 'disco', 'classical', 'hiphop', 'jazz','country', 'pop', 'blues', 'reggae', 'rock']

root = tkinter.Tk()
root.withdraw()
audio_path = askopenfilename()

signal, sr = librosa.load(audio_path)
melspec = librosa.feature.melspectrogram(signal[:660000], sr=sr, n_fft=2048, hop_length=512).T[:128, ]

model = load_model('gtzan_exec_10_epochs_200.h5')

X_train = []
X_train.append(melspec)
X_train = np.array(X_train)
predictions = model.predict(X_train)
y_classes = predictions.argmax(axis=-1)
print(y_classes, genres[y_classes[0]])

plt.figure()
plt.title('Predictions')
plt.ylabel('Prediction')
plt.xlabel('Genres')
plt.bar(genres, predictions[0])
plt.show()

from __future__ import print_function

# We'll need numpy for some mathematical operations
import numpy as np

# matplotlib for displaying the output
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

# And the display module for visualization
import librosa.display
import tkinter
from tkinter.filedialog import askopenfilename

root = tkinter.Tk()
root.withdraw()
audio_path = askopenfilename()

print(audio_path)
# splitted = audio_path.split("/")
# print(splitted[-1], splitted[-2])

y, sr = librosa.load(audio_path)

# Let's make and display a mel-scaled power (energy-squared) spectrogram
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# Convert to log scale (dB). We'll use the peak power (max) as reference.
log_S = librosa.power_to_db(S, ref=np.max)

# Make a new figure
plt.figure(figsize=(12,4))

librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

plt.show()

# plt.savefig('fig-1')
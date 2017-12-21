from __future__ import print_function

# We'll need numpy for some mathematical operations
import numpy as np

# matplotlib for displaying the output
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
#matplotlib inline

# and IPython.display for audio output
import IPython.display

# Librosa for audio
import librosa
# And the display module for visualization
import librosa.display
import tkinter
from tkinter.filedialog import askopenfilename
import os

root = tkinter.Tk()
root.withdraw()

def trainingsetToSpectogram(input_path, output_path):
    # Structure for the array of songs
    genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
      'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
    song_data = []
    genre_data = []
    files_array = []

    # Read files from the folders
    for x, _ in genres.items():
        for root, subdirs, files in os.walk(input_path + x):
            for file in files:
                print(x, file)
                # Read the audio file
                file_name = input_path + x + "/" + file
                files_array.append(file)

                y, sr = librosa.load(file_name)

                # Let's make and display a mel-scaled power (energy-squared) spectrogram
                S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

                # Convert to log scale (dB). We'll use the peak power (max) as reference.
                log_S = librosa.power_to_db(S, ref=np.max)

                # Make a new figure
                plt.figure(figsize=(12, 4))

                librosa.display.specshow(log_S, sr=sr)

                # plt.show()
                filename = file.split('.')

                plt.savefig('output/' + x + '/' + filename[0] + filename[1])
                plt.close()

                # print(file_name)
                # exit()
    return files_array

print(trainingsetToSpectogram('../genre-classification/gtzan/', '../genre-classification/output/'))
print("done")

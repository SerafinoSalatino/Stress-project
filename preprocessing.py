import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import librosa as lr
import librosa.display
import librosa.feature
import os
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

DIR = "Audio_Speech_Actors_01-24"
y = []
x = []
for dir in tqdm(os.listdir('./'+ DIR)):
    audio_files = glob('./' + DIR + '/' + dir + '/*.wav')
    for file in audio_files:
        val = file.split('-')[3] 
        if ( val == '01' or val == '02' or val == '03'):
            y.append(0)
        elif ( val == '08'):
            continue
        else:
            y.append(1)
        audio,sr = lr.load(file)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        #feature audio

        rms = np.mean(librosa.feature.rms(audio)) #prima feature
        zc = np.sum(librosa.zero_crossings(audio)) #seconda feature
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio)) #terza feature
        sc = np.mean(librosa.feature.spectral_centroid(audio,sr=sr)) #quarta feature
        sro = np.mean(librosa.feature.spectral_rolloff(audio,sr=sr)) #quinta feature
        sbw = np.mean(librosa.feature.spectral_bandwidth(audio,sr=sr)) #sesta feature
        chroma = np.mean(librosa.feature.chroma_stft(audio, sr=sr)) #settima feature
        mfcc = librosa.feature.mfcc(audio,sr=sr,n_mfcc=40)  #40 feature di mfcc

        delta = np.mean(librosa.feature.delta(mfcc)) #ottava feature
        pitch = np.mean(pitches) #nona feature
        max_amplitude = np.max(audio)


        a = [rms,zc,zcr,sc,sro,sbw,chroma,delta,pitch,max_amplitude]
        for s in mfcc:
            a.append(np.mean(s))
        x.append(a)
np.save("array_labels.npy",y)
np.save("array_dataset.npy",x)
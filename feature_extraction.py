from pydub import AudioSegment
import pandas as pd
import numpy as np
import librosa
import os
import sys

print("Done importing")

class AudioExtractor:

    def __init__(self, file):
        self.file = file
        self.features = dict(
            chroma_stft=12,
            chroma_cqt=12,
            # chroma_cens=1,
            # spectral_centroid=1,
            spectral_bandwidth=1,
            spectral_contrast=1,
            spectral_rolloff=1,
            mfcc=12, 
            # rmse=1, 
            # zcr=1
        )
        self.stats = (
            'mean', 
            'std',
            # 'median',
            # 'min',
            # 'max',
        )
        
    def read_audio(self, duration):
        data, sampling_rate = librosa.load(self.file, res_type='kaiser_fast', duration=duration)
        return data, sampling_rate
        
    def count_stats(self, data):
        all_stats = {}
        for stat in self.stats:
            if stat == 'mean':
                all_stats[stat] = np.mean(data, axis=1)
            elif stat == 'std':
                all_stats[stat] = np.std(data, axis=1)
            elif stat == 'median':
                all_stats[stat] = np.median(data, axis=1)
            elif stat == 'min':
                all_stats[stat] = np.min(data, axis=1)
            elif stat == 'max':
                all_stats[stat] = np.max(data, axis=1)
        return all_stats
        
    def extract(self, duration=3.0):
        data, sampling_rate = self.read_audio(duration)
        S = np.abs(librosa.stft(data))
        fixed_features = {}
        for feature in self.features:
            coeff = self.features[feature]
            counted_stats = None
            if feature == "chroma_stft":
                counted_stats = self.count_stats(librosa.feature.chroma_stft(y=data, sr=sampling_rate, n_chroma=coeff))
            elif feature == "chroma_cqt":
                counted_stats = self.count_stats(librosa.feature.chroma_cqt(y=data, sr=sampling_rate, n_chroma=coeff))
            elif feature == "chroma_cens":
                counted_stats = self.count_stats(librosa.feature.chroma_cens(y=data, sr=sampling_rate, n_chroma=coeff))
                
            elif feature == "spectral_centroid":
                counted_stats = self.count_stats(librosa.feature.spectral_centroid(y=data, sr=sampling_rate))
            elif feature == "spectral_bandwidth":
                counted_stats = self.count_stats(librosa.feature.spectral_bandwidth(y=data, sr=sampling_rate))
            elif feature == "spectral_contrast":
                counted_stats = self.count_stats(librosa.feature.spectral_contrast(S=S, sr=sampling_rate))
            elif feature == "spectral_rolloff":
                counted_stats = self.count_stats(librosa.feature.spectral_rolloff(y=data, sr=sampling_rate))
                
            elif feature == "mfcc":
                counted_stats = self.count_stats(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=coeff))
            elif feature == "rmse":
                counted_stats = self.count_stats(librosa.feature.rmse(y=data))
            elif feature == "zcr":
                counted_stats = self.count_stats(librosa.feature.zero_crossing_rate(y=data))
            for counted_stat in counted_stats:
                for i in range(len(counted_stats[counted_stat])):
                    value = counted_stats[counted_stat][i]
                    fixed_features[feature+"_"+counted_stat+"_"+str(i+1)] = [value]
        return fixed_features


def convert_to_wav(file, output_path):
    song = AudioSegment.from_mp3(file)
    first = song[:10*1000]
    last = song[-10*1000:]
    song = first + last
    song.export(output_path + file.replace(".mp3", ".wav"), format="wav")
    
def convert_audios_to_wav(path):
    """
        path    : path that contains mp3 audio files
        return  : new directory contains wav format
    """
    directory = "./converted/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    files = [x for x in os.listdir(path) if ".mp3" in x]
    counter = 1
    print("converting",len(files),"files")
    for file in files:
        if ".mp3" in file:
            convert_to_wav(file, directory)
            print(counter,"files converted")
            counter += 1
    return directory
    
def generate_csv(path, output, format, is_mp3=False, label=""):
    """
        extract all audio feature inside path to csv file
        path    : path that contains audio files
        output  : csv output file name
        format  : .wav or .au
        label   : genre of audio files
        is_mp3  : is audio in mp3 format or not
        
    """
    if is_mp3:
        path = convert_audios_to_wav(path)
    files = [path + x for x in os.listdir(path) if format in x]
    
    m = len(files)
    features = AudioExtractor(files[0]).extract()
    features["Target"] = label
    df = pd.DataFrame(features)
    print(1, "/", m, "files extracted")
    for i in range(1, m):
        print(i+1, "/", m, "files extracted")
        features = AudioExtractor(files[i]).extract()
        features["Target"] = label
        temp = pd.DataFrame(features)
        df = pd.concat([df, temp])
    df.to_csv(output, index=False)

def start():
    root = './genres/'
    genres = [_ for _ in os.listdir(root)]
    for genre in genres:
        label = genre
        dir = root + genre + "/"
        generate_csv(dir, label + '.csv', 'au', label=label)
    
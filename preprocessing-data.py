import os 
import math
import numpy 
import librosa
import json
import tensorflow as tf
import argparse

from sklearn.preprocessing import normalize, MinMaxScaler
from scipy import misc

MFCC_FEATURES = 13
NUM_FEATURES = 1
N_FFT = 2048
HOP_LENGTH = 512
SONG_LENGTH = 30
SAMPLE_RATE = 22050
SAMPLE_PER_TRACK = SAMPLE_RATE*SONG_LENGTH
NUM_SEGMENTS = 10

def get_filenum(path, genres):
    #returns the number of files given the file path and classes
    num_files = 0

    for i in genres:
        file = os.listdir(path + '/' + i)
        for j in file :
            if j.split('.')[-1] == 'wav':
                num_files +=1
                
    return num_files


def get_input_len(path, genres):
    # returns the input length of the audio file
    folder = genres[0]
    files = os.listdir(path+'/'+folder)
    for i in files:
        if i.split('.')[-1] == "wav":
            sample, sr = librosa.load(path+'/'+folder+'/'+i)
            input_length = len(sample) 
            break
        
    return input_length

def get_labels(path):
    # gets the class labels from the file path
    labels = []
    for i in os.scandir(path = path):
        if i.is_dir():
            labels.append(i.name)
    return labels


def load_data(path, sr):
    # loads an audio signal into a numpy array from a file path
    signal, sample_rate = librosa.load(path, sr=sr)
    return signal, sample_rate

def mfcc_conv(data,sample_rate,  num_mfcc, n_fft, hop_length):
    # extracts MFCC features from an audio signal and converts them
    mfcc = librosa.feature.mfcc(y = data, sr = sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T
    return mfcc
         
def create_json(json_path, data):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    return 0

def create_mfcc_json(audio_path,json_path, num_mfcc, n_fft, hop_length, split_num):
    # creates a json file containing labels and MFCC features per audio signal
    data = {
        "Genres": [],
        "labels": [],
        "mfcc_features": []
    }
    labels = get_labels(audio_path)   
    for i in labels:
        data["Genres"].append(i)
        
    samples_in_segment = int(SAMPLE_PER_TRACK / split_num)
    mfcc_count_in_segment = math.ceil(samples_in_segment/hop_length)
    #iterates through every folder and file in the directory 
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(audio_path)):
        print(f"processing folder {labels[i-1]}")
        
        for j in filenames:      
            #loads the signals for every file, splits them into a 1/10th segments and extracts MFCC features which are then written to teh JSON file   
            path = os.path.join(dirpath, j)
            signal,sr = load_data(path,sr = SAMPLE_RATE)
            for k in range(NUM_SEGMENTS):
                start = samples_in_segment * k
                end = start + samples_in_segment
                mfcc = mfcc_conv(signal[start:end],sr,num_mfcc,n_fft,hop_length)
                if len(mfcc) == mfcc_count_in_segment:
                    data["mfcc_features"].append(mfcc.tolist())
                    data["labels"].append(i-1)
                    print(f"{path}, segment: {k+1}")

    #writes the data object to a json file                    
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
def create_aug_mfcc_json(audio_path,json_path, num_mfcc, n_fft, hop_length, split_num):
    #the same as the previous function but adds pitch shifted features aswell
    data = {
        "Genres": [],
        "labels": [],
        "mfcc_features": []
    }
    labels = get_labels(audio_path)   
    for i in labels:
        data["Genres"].append(i)
        
    samples_in_segment = int(SAMPLE_PER_TRACK / split_num)
    mfcc_count_in_segment = math.ceil(samples_in_segment/hop_length)
    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(audio_path)):
        print(f"processing folder {labels[i-1]}")
        
        for j in filenames:         
            path = os.path.join(dirpath, j)
            signal,sr = load_data(path,sr = SAMPLE_RATE)
            for k in range(NUM_SEGMENTS):
                start = samples_in_segment * k
                end = start + samples_in_segment

                mfcc = mfcc_conv(signal[start:end],sr,num_mfcc,n_fft,hop_length)
                shifted_mfcc = mfcc_conv((librosa.effects.pitch_shift(y= signal[start:end],sr= sr, n_steps= 4)),sr,num_mfcc,n_fft,hop_length)
                if len(mfcc) == mfcc_count_in_segment & len(shifted_mfcc) == mfcc_count_in_segment:
                    data["mfcc_features"].append(mfcc.tolist())
                    data["labels"].append(i-1)
                    data["mfcc_features"].append(shifted_mfcc.tolist())
                    data["labels"].append(i-1)
                    print(f"{path}, segment: {k+1}")

            
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple script that creates a JSON file')
    parser.add_argument('path', type=str, help='the path to the dataset')
    parser.add_argument('-d', '--data', choices=['regular', 'augmented'], default='regular', help='type of data to create')
    parser.add_argument('filename', type=str,default = "features.json",help='name of the json file being written')

    args = parser.parse_args()
    
    if args.data == 'regular':
        print("creating regular mfcc file")
        create_mfcc_json(args.path, args.filename, MFCC_FEATURES,N_FFT,HOP_LENGTH,NUM_SEGMENTS)
    elif args.data == 'augmented':
        print('creating augmented data mfcc file')
        create_aug_mfcc_json(args.path, args.filename, MFCC_FEATURES, N_FFT, HOP_LENGTH, NUM_SEGMENTS)
    else:
        raise Exception("incorrect keyword")
import json
import numpy
import tensorflow
import tensorflow.keras as keras
from keras.layers import Conv2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import os
import librosa
import math

MFCC_FEATURES = 13
N_FFT = 2048
HOP_LENGTH = 512
SONG_LENGTH = 30
SAMPLE_RATE = 22050
SAMPLE_PER_TRACK = SAMPLE_RATE*SONG_LENGTH
NUM_SEGMENTS = 10

classes = [
    "blues:0",
    "classical:1",
    "country:2",
    "disco:3",
    "hiphop:4",
    "jazz:5",
    "metal:6",
    "pop:7",
    "reggae:8",
    "rock:9"
]

classesX = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9
}

def load_features(mfcc_path):
    #loads the data and labels from the JSON file
    with open(mfcc_path,'r') as f:
        data = json.load(f)
              
    x_data = numpy.array(data["mfcc_features"])
    y_labels = numpy.array(data["labels"])
    
    return x_data, y_labels


def add_axis(arr):
    #adds a dimension to an array 
    arr = arr[..., numpy.newaxis]
    return arr

def split_data(data, labels, testing_size, validation_size, model_name):
    training_data , testing_data, training_labels, testing_labels = train_test_split(data, labels, test_size= testing_size)
    training_data, validation_data, training_labels, validation_labels = train_test_split(training_data, training_labels, test_size= validation_size)
    if (model_name == 'cnn'):
        training_data = add_axis(training_data)
        testing_data = add_axis(testing_data)
        validation_data = add_axis(validation_data)
        return training_data, validation_data, testing_data, training_labels, validation_labels, testing_labels
    elif ( model_name == 'lstm'):
        return training_data, validation_data, testing_data, training_labels, validation_labels, testing_labels
    else:
        print("this model does not exist ")
        return 0 

def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="training accuracy")
    axs[0].plot(history.history["val_accuracy"], label="testing accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")


    plt.show()
    
def cnn_model(input_shape,class_num):
    #CNN model with 3 convolutoinal layers and a fully connected layer
    model = keras.Sequential()
    
    model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = input_shape))
    model.add(keras.layers.MaxPooling2D((3,3), strides = (2,2), padding = 'same'))
    model.add(keras.layers.BatchNormalization())

    
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    
    
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())


    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(class_num, activation='softmax'))
    
    return model

def lstm_model(input_shape, class_num):
    #LSTM Layer with 2 LSTM layers and a fully connected layer
    model = keras.Sequential()
    
    model.add(keras.layers.LSTM(128, input_shape = input_shape, return_sequences = True))
    model.add(keras.layers.LSTM(64))
    model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Dense(64, activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Dense(class_num, activation = 'softmax'))
    
    return model


def predict(model, audio_input, label):
    audio_input = audio_input[numpy.newaxis,...]
    #uses keras predict function to get a probability over 10 nodes
    prediction = model.predict(audio_input)
    #converts the output to an integer for the node with the highest probablity
    predicted_label = numpy.argmax(prediction, axis = 1)
    
    return prediction, predicted_label, label
    
def create_input_shape(data, model_name):
    #adds an extra axis to the cnn input shape but leaves the lstm one as is
    if (model_name == 'cnn'):
        input_shape = (data.shape[1], data.shape[2], 1)
        return input_shape
    elif (model_name == 'lstm'):
        input_shape = (data.shape[1], data.shape[2])
        return input_shape
    else:
        print("model not found")
        return 0

def compile_model(model, lr):
    #usese keras to optimize the model with Adam and uses a sparse_categorical_crossentropy loss function
    optimiser = keras.optimizers.Adam(learning_rate = lr)
    model.compile(optimizer = optimiser,
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy']
                  )
    model.summary()
    return model


def get_label(path):
    # returns the class of a file given its path
    folder_name = os.path.basename(os.path.dirname(path))
    return folder_name

def mfcc_conv(data,sample_rate,  num_mfcc, n_fft, hop_length):
    #converts a signal into an mfcc file
    mfcc = librosa.feature.mfcc(y = data, sr = sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T
    return mfcc

def prep_test_data(path):
    #prepares input data similarly to how its done in the data_preprocessing.py file
    input_data = {
        'mfcc_features' : [],
        'labels' : []
    }
    signal, sr = librosa.load(path= path)
    label = get_label(path)
    print(signal, sr)
    samples_in_segment = int(SAMPLE_PER_TRACK / NUM_SEGMENTS)
    mfcc_count_in_segment = math.ceil(samples_in_segment/HOP_LENGTH)
    for k in range(NUM_SEGMENTS):
        start = samples_in_segment * k
        end = start + samples_in_segment
        mfcc = mfcc_conv(signal[start:end],sr,MFCC_FEATURES,N_FFT,HOP_LENGTH)
        if len(mfcc) == mfcc_count_in_segment :
            input_data["mfcc_features"].append(mfcc.tolist())
            input_data["labels"].append(label)
    #returns a numpy array of the 13 mfcc features taken from 10 segments and a normal array holding the labels of the data
    input_x = numpy.array(input_data['mfcc_features'])
    input_y =input_data['labels']
    
            
    return input_x, input_y




if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='A simple script that creates a JSON file')
    parser.add_argument('path', type=str, help='path for the feature JSON file')
    parser.add_argument('-p','--prediction', type = str, help = 'path to the prediction')
    parser.add_argument('-m', '--model', choices=['cnn', 'lstm'], default='cnn', help='choice of models')
    parser.add_argument('-v', '--verbose', action = 'store_true', help= 'print verbose output')

    args = parser.parse_args()
    
    path = args.path
    data, labels = load_features(path)
    
    if args.model == 'cnn': 
        print('starting cnn program...')
        
        training_data, validation_data, testing_data, training_labels, validation_labels, testing_labels = split_data(data, labels, 0.25, 0.2, 'cnn')
        input_shape = create_input_shape(training_data,'cnn')
        
        model = cnn_model(input_shape, 10)
        model = compile_model(model, 0.0001)
        history = model.fit(training_data, training_labels, validation_data = (validation_data, validation_labels), batch_size = 32, epochs = 30)
        if args.verbose:

            plot_history(history)
            
            test_loss, test_accuracy = model.evaluate(testing_data, testing_labels, verbose = 2)
            print(f"the accuracy on training data is: {test_accuracy}")
        
        if args.prediction:
            print('using input data')
            x,y = prep_test_data(args.prediction)
            #uses the middle segment from input data as in genral it holds the most information about a song
            predict_x = add_axis(x[5])
            predict_y = y[5]
        else:
            predict_x = testing_data[190]
            predict_y = testing_labels[190]

        
        prediction, predicted_label, label =predict(model, predict_x, predict_y)
        print(classes)
        print(f"predicted label {predicted_label}, actual_label {label}")
    elif args.model == 'lstm':
        print('starting lstm program...')
        training_data, validation_data, testing_data, training_labels, validation_labels, testing_labels = split_data(data, labels, 0.25, 0.2, 'lstm')
        input_shape = create_input_shape(training_data,'lstm')
        
        model = lstm_model(input_shape, 10)
        model = compile_model(model, 0.0001)
        history = model.fit(training_data, training_labels, validation_data = (validation_data, validation_labels), batch_size = 32, epochs = 10)
        if args.verbose:
            plot_history(history)        
            test_loss, test_accuracy = model.evaluate(testing_data, testing_labels, verbose = 2)
            print(f"the accuracy on training data is: {test_accuracy}")
            
        if args.prediction:
            print('using input data')
            x,y = prep_test_data(args.prediction)
            predict_x = add_axis(x[5])
            predict_y = y[5]
        else:
            predict_x = testing_data[190]
            predict_y = testing_labels[190]   
            
        prediction, predicted_label, label =predict(model, predict_x, predict_y)
        print(classes)
        print(f"predicted label {predicted_label}, actual_label {label}")
    
        
        
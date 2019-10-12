from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import os
from PIL import Image
import multiprocessing
import librosa
import numpy as np
import math
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.optimizers import Adadelta, Adam
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import BatchNormalization, Input, GaussianNoise
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
DEBUG = True
train_models = 1
LOAD_WEIGHTS_CNN = 0
LOAD_WEIGHTS_RCNN = 0
LOAD_WEIGHTS_VGG = 0
VAL_SPLIT = 0.2
COL_SIZE = 173
BATCH_SIZE1 = 64
BATCH_SIZE2 = 64
BATCH_SIZE3 = 128
N_FOLDS = 3
EPOCHS1 = 35 #35#70
EPOCHS2 = 35 #35#70
EPOCHS3 = 35
plot_layer_visualizations_flag = 0
save_accuracy_and_loss_plots = 0
save_predictions = 0

RS = 0
SILENCE_THRESHOLD = .01
RATE = 22050
N_MFCC = 3


print("N_MFCC ",N_MFCC," Col_size: ",COL_SIZE," Epochs: ",EPOCHS1)

def to_categorical(y):
    '''
    Converts list of languages into a binary class matrix
    :param y (list): list of languages
    :return (numpy array): binary class matrix
    '''
    g = []

    for i in range(0,len(y)):
        if y[i]==0:
            g.append([1,0,0])
        elif y[i] == 1:
            g.append([0,1,0])
        elif y[i] == 2:
            g.append([0,0,1])
    
    return g

def to_single(y):
    '''
    Converts list of languages into a binary class matrix
    :param y (list): list of languages
    :return (numpy array): binary class matrix
    '''
    g = []

    for i in range(0,len(y)):
        if y[i]==[1,0,0]:
            g.append(0)
        if y[i] == [0,1,0]:
            g.append(1)
        if y[i] == [0,0,1]:
            g.append(2)
    
    return g

def to_mfcc(wav):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    :param wav (numpy array): Wav form
    :return (2d numpy array: MFCC
    '''
    return(librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC))

def remove_silence(wav, thresh=0.04, chunk=5000):
    '''
    Searches wav form for segments of silence. If wav form values are lower than 'thresh' for 'chunk' samples, 
     the values will be removed
    :param wav (np array): Wav array to be filtered
    :return (np array): Wav array with silence removed
    '''

    tf_list = []
    for x in range(len(wav) / chunk):
        if (np.any(wav[chunk * x:chunk * (x + 1)] >= thresh) or np.any(wav[chunk * x:chunk * (x + 1)] <= -thresh)):
            tf_list.extend([True] * chunk)
        else:
            tf_list.extend([False] * chunk)

    tf_list.extend((len(wav) - len(tf_list)) * [False])
    return(wav[tf_list])

def normalize_mfcc(mfcc):
    '''
    Normalize mfcc
    :param mfcc:
    :return:
    '''
    mms = MinMaxScaler()
    return(mms.fit_transform(np.abs(mfcc)))

def make_segments(mfccs,labels):
    '''
    Makes segments of mfccs and attaches them to the labels
    :param mfccs: list of mfccs
    :param labels: list of labels
    :return (tuple): Segments with labels
    '''
    segments = []
    seg_labels = []
    for mfcc,label in zip(mfccs,labels):
        for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
            segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
            seg_labels.append(label)
    return(segments, seg_labels)

def segment_one(mfcc):
    '''
    Creates segments from on mfcc image. If last segments is not long enough to be length of columns divided by COL_SIZE
    :param mfcc (numpy array): MFCC array
    :return (numpy array): Segmented MFCC array
    '''
    segments = []
    for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
        segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
    return(np.array(segments))

def create_segmented_mfccs(X_train):
    '''
    Creates segmented MFCCs from X_train
    :param X_train: list of MFCCs
    :return: segmented mfccs
    '''
    segmented_mfccs = []
    for mfcc in X_train:
        segmented_mfccs.append(segment_one(mfcc))
    return(np.array(segmented_mfccs))


def predict_prob_class_audio(MFCCs, model):
    '''
    Predict class based on MFCC samples' probabilities
    :param MFCCs: Numpy array of MFCCs
    :param model: Trained model
    :return: Predicted class of MFCC segment group
    '''
    MFCCs = MFCCs.reshape(MFCCs.shape[0],MFCCs.shape[1],MFCCs.shape[2],1)
    y_predicted = model.predict_proba(MFCCs,verbose=0)
    return (np.argmax(np.sum(y_predicted,axis=0)))

def predict_class_audio(MFCCs, model):
    '''
    Predict class based on MFCC samples
    :param MFCCs: Numpy array of MFCCs
    :param model: Trained model
    :return: Predicted class of MFCC segment group
    '''
    
    MFCCs = MFCCs.reshape(1,MFCCs.shape[0],MFCCs.shape[1],MFCCs.shape[2])
    y_predicted = model.predict_classes(MFCCs,verbose=0)
    return (Counter(list(y_predicted)).most_common(1)[0][0])

def predict_class_all(X_train, model):
    '''
    :param X_train: List of segmented mfccs
    :param model: trained model
    :return: list of predictions
    '''
    def predict_class_audio(MFCCs, model):
        '''
        Predict class based on MFCC samples
        :param MFCCs: Numpy array of MFCCs
        :param model: Trained model
        :return: Predicted class of MFCC segment group
        '''
        
        MFCCs = MFCCs.reshape(1,MFCCs.shape[0],MFCCs.shape[1],MFCCs.shape[2])
        y_predicted = model.predict_classes(MFCCs,verbose=0)
        return (Counter(list(y_predicted)).most_common(1)[0][0])
    predictions = []
    for mfcc in X_train:
        predictions.append(predict_class_audio(mfcc, model))
        # predictions.append(predict_prob_class_audio(mfcc, model))
    return predictions

def confusion_matrix(y_predicted,y_test):
    '''
    Create confusion matrix
    :param y_predicted: list of predictions
    :param y_test: numpy array of shape (len(y_test), number of classes). 1.'s at index of actual, otherwise 0.
    :return: numpy array. confusion matrix
    '''
    confusion_matrix = np.zeros((3,3),dtype=int )
    for index, predicted in enumerate(y_predicted):
        confusion_matrix[np.argmax(y_test[index])][predicted] += 1
    return(confusion_matrix)

def get_accuracy(y_predicted,y_test):
    '''
    Get accuracy
    :param y_predicted: numpy array of predictions
    :param y_test: numpy array of actual
    :return: accuracy
    '''
    c_matrix = confusion_matrix(y_predicted,y_test)
    return( np.sum(c_matrix.diagonal()) / float(np.sum(c_matrix)))


def get_callbacks(name_weights, patience_lr, batch_size):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    es = EarlyStopping(monitor='val_loss', patience= patience_lr + 3, verbose=0, mode='min')

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=True,
                    write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                    embeddings_metadata=None)

    return [mcp_save, reduce_lr_loss, es, tb]

def create_cnn_model(input_shape, num_classes,load_weights = 0, string =""):
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(32, kernel_size = (3, 3), strides = (1,1), padding = 'same', activation = 'relu', 
                        data_format="channels_last", input_shape=input_shape,name="Conv2d1"))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (4, 4),name="MPool2d1"))
    classifier.add(Dropout(rate = 0.3))

    classifier.add(Conv2D(64,kernel_size=(3,3), strides = (1,1), padding = 'same', activation='relu',name="Conv2d2"))
    classifier.add(MaxPooling2D(pool_size=(2, 2),name="MPool2d2"))
    classifier.add(Dropout(rate = 0.3))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(activation = 'relu', units = 128,name="FC128d1"))
    classifier.add(Dropout(rate = 0.3))
    classifier.add(Dense(activation = 'softmax', units = num_classes,name="output_layer"))
    if load_weights == 1:    
        classifier.load_weights("models/cnn_model_weights"+string+".best.hdf5")
    # Compiling the CNN
    classifier.compile(loss = 'categorical_crossentropy',
                    optimizer = 'adam', 
                    metrics = ['accuracy'])
    return classifier

def create_rcnn_model(input_shape, num_classes, load_weights = 0, string =""):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3,3), strides = (1,1), padding = 'same', activation='relu', 
                        data_format="channels_last", input_shape=input_shape,name="Conv2d1"))
        model.add(MaxPooling2D(pool_size=(4, 4),name="MPool2d1"))
        model.add(Dropout(rate = 0.3))

        model.add(Conv2D(64,kernel_size=(3,3), strides = (1,1), padding = 'same', activation='relu',name="Conv2d2"))
        model.add(MaxPooling2D(pool_size=(3, 3),name="MPool2d2"))
        model.add(Dropout(rate = 0.3))
        model.add(Flatten())
        model.add(Dense(activation='relu', units = 128, name="FC128d1"))
        model.add(Dropout(rate = 0.3))
        model.add(Dense(activation='softmax', units = num_classes,name="output_layer"))
        if load_weights == 1:
            model.load_weights("models/rcnn_model_weights"+string+".best.hdf5")
    
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        return model

def create_vgg_model(input_shape, num_classes, load_weights = 0, string =""):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3,3), strides = (1,1), padding = 'same', activation='relu', 
                        data_format="channels_last", input_shape=input_shape,name="Conv2d1"))
        model.add(MaxPooling2D(pool_size=(3, 3),name="MPool2d1"))
        model.add(Dropout(rate = 0.3))
        model.add(Conv2D(64,kernel_size=(3,3), strides = (1,1), padding = 'same', activation='relu',name="Conv2d2"))
        model.add(MaxPooling2D(pool_size=(3, 3),name="MPool2d2"))
        model.add(Dropout(rate = 0.3))        
        model.add(Conv2D(64,kernel_size=(3,3), strides = (1,1), padding = 'same', activation='relu',name="Conv2d3"))
        model.add(MaxPooling2D(pool_size=(2, 2),name="MPool2d3"))
        model.add(Flatten())
        model.add(Dropout(rate = 0.3))
        model.add(Dense(activation='relu', units = 128, name="FC128d1"))
        model.add(Dense(activation='softmax', units = num_classes, name="output_layer"))
        if load_weights == 1:
            model.load_weights("models/vgg_model_weights"+string+".best.hdf5")
    
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        return model

def get_layer_outputs(image):
    test_image = image
    outputs    = [layer.output for layer in model.layers]          # all layer outputs
    comp_graph = [K.function([model.input]+ [K.learning_phase()], [output]) for output in outputs]  # evaluation functions

    # Testing
    layer_outputs_list = [op([test_image, 1.]) for op in comp_graph]
    layer_outputs = []

    for layer_output in layer_outputs_list:
        print(layer_output[0][0].shape, end='\n-------------------\n')
        layer_outputs.append(layer_output[0][0])

    return layer_outputs

def plot_layer_outputs(layer_number):    
    layer_outputs = get_layer_outputs()

    x_max = layer_outputs[layer_number].shape[0]
    y_max = layer_outputs[layer_number].shape[1]
    n     = layer_outputs[layer_number].shape[2]

    L = []
    for i in range(n):
        L.append(np.zeros((x_max, y_max)))

    for i in range(n):
        for x in range(x_max):
            for y in range(y_max):
                L[i][x][y] = layer_outputs[layer_number][x][y][i]


    for img in L:
        plt.figure()
        plt.imshow(img, interpolation='nearest')


def no_split_cnn_model(X, y, X_subm, batch_size=64, load_weights = 0, n_folds = N_FOLDS):
    # Get row, column, and class sizes
    rows = X[0].shape[0]
    cols = X[0].shape[1]
    val_rows = X_subm[0].shape[0]
    val_cols = X_subm[0].shape[1]
    num_classes = 3
    # input image dimensions to feed into 2D ConvNet Input layer
    input_shape = (rows, cols, 1)

    # Stops training if accuracy does not change at least 0.005 over 10 epochs
    rlrl = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=7, verbose=1, min_delta=1e-4, mode='min')
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=True,
                    write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                    embeddings_metadata=None)
    classifier = create_cnn_model(input_shape = input_shape, num_classes=num_classes, load_weights = load_weights)
    mcp_save = ModelCheckpoint("models/cnn_model_weights.best.hdf5", save_best_only=True, monitor='val_loss', mode='min')
    # Image shifting
    train_datagen = ImageDataGenerator(width_shift_range=0.05)
    train_datagen.fit(X)
    print("Count of y class labels (should be even)",Counter(np.array(y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RS, stratify=y)
    count_y_train = Counter(y_train)
    count_y_test = Counter(y_test)

    print("Count of y_train class labels (should be even)",count_y_train)
    print("Count of y_test class labels (should be even)",count_y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # Fit model using ImageDataGenerator
    train_model = classifier.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                                steps_per_epoch = math.ceil(len(X_train) / batch_size),
                                epochs = EPOCHS1,
                                validation_steps = math.ceil(len(X_test) / batch_size), 
                                verbose = 1,
                                callbacks=[es,tb, mcp_save, rlrl], 
                                validation_data=(train_datagen.flow(X_test,y_test,batch_size = batch_size)),
                                shuffle = True)
    print("Average Accuracy: ", round(np.mean(train_model.history['acc']),4), "% +/- ", round(np.std(train_model.history['acc']),2), "% (Max = ",round(np.max(train_model.history['acc']),4),"%)")
    print("Average Validation Accuracy: ",round(np.mean(train_model.history['val_acc']),4), "% +/- ", round(np.std(train_model.history['val_acc']),2), "% (Max = ",round(np.max(train_model.history['val_acc']),4),"%)")
    print("Average Loss: ", round(np.mean(train_model.history['loss']),4), " +/- ", round(np.std(train_model.history['loss']),2), " (Min = ",round(np.min(train_model.history['loss']),4),")")
    print("Average Validation Loss: ",round(np.mean(train_model.history['val_loss']),4), " +/- ", round(np.std(train_model.history['val_loss']),2), " (Min = ",round(np.min(train_model.history['val_loss']),4),")")
    print("Min Loss: ", round(np.min(train_model.history['loss']),4))
    print("Min Validation Loss: ", round(np.min(train_model.history['val_loss']),4))   
    print(classifier.summary())
    print("CNN Model Output ^^^, model fit complete===========================")
    if save_accuracy_and_loss_plots == 1:
        print("Plotting Accuracy and Loss Curves." )

    
        plt.figure(1)
        plt.plot(train_model.history['acc'], label = "Train Acc.",marker='', color='skyblue', linewidth=4)
        plt.plot(train_model.history['val_acc'], label = "Validation Acc.", marker='', color='crimson', linewidth=2)
        plt.title('CNN Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(('Train Acc.','Validation Acc.'))
        plt.savefig('cnn_model_accuracy.png')    
        plt.close()

        plt.figure(2)
        plt.plot(train_model.history['loss'], label = "Train Loss",marker='', color='skyblue', linewidth=4)
        plt.plot(train_model.history['val_loss'], label = "Validation Loss", marker='', color='salmon', linewidth=2)
        plt.title('CNN Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(('Train Loss','Validation Loss'))
        plt.savefig('cnn_model_loss.png')       
        plt.close()
    #train_model output  
    print("=================================================================")    
    print("cnn_model model output ========================================")
    print("=================================================================")                    
    y_predicted_1 = classifier.predict_classes(X_test)        
    # Print statistics


    print("Count of Predictions from cnn_model on  test data: \n", Counter(y_predicted_1))
    print('Confusion matrix of test samples(X_test, cnn_model):\n', np.sum(confusion_matrix(y_predicted_1, y_test),axis=0))
    print('Confusion matrix :\n',confusion_matrix(y_predicted_1, y_test))
    print('Accuracy :', round(get_accuracy(y_predicted_1, y_test),4))
    test_datagen = ImageDataGenerator(width_shift_range=0.05)
    y_predictions_cnn = classifier.predict_classes(X_subm)
    y_prob_cnn = classifier.predict_proba(X_subm)
    print("cnn_model count of predictions, submission data: ",Counter(y_predictions_cnn))
    y_subm_cnn = Counter(y_predictions_cnn)
    print("cnn_model first 10 predictions: ",y_predictions_cnn[:10])
    if save_predictions == 1:
        L = list(range(5377))
        prediction = pd.DataFrame({'file_id':[20000+c for c in L],
                                    'accent': y_predictions_cnn,
                                    'probs_0': [elem[0] for elem in y_prob_cnn],
                                    'probs_1': [elem[1] for elem in y_prob_cnn],
                                    'probs_2': [elem[2] for elem in y_prob_cnn]}).to_csv('prediction1.csv',index=False)

    return classifier, y_subm_cnn, y_predictions_cnn, y_prob_cnn

def no_split_rcnn_model(X, y, X_subm, batch_size=64, load_weights = 0, n_folds = N_FOLDS): #64
    '''
    Trains 2D convolutional neural network
    :param X_train: Numpy array of mfccs
    :param y_train: Binary matrix based on labels
    :return: Trained model
    '''
    # Get row, column, and class sizes
    rows = X[0].shape[0]
    cols = X[0].shape[1]
    val_rows = X_subm[0].shape[0]
    val_cols = X_subm[0].shape[1]
    num_classes = 3
    # input image dimensions to feed into 2D ConvNet Input layer
    input_shape = (rows, cols, 1)
    
    # Stops training if accuracy does not change at least 0.005 over 10 epochs
    
    rlrl = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=7, verbose=1, min_delta=1e-4, mode='min')
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size, write_graph=True  , write_grads=True,
                    write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                    embeddings_metadata=None)
    model = create_rcnn_model(input_shape = input_shape, num_classes=num_classes, load_weights = load_weights)    
    model.save_weights("models/rcnn_model_weights.best.hdf5", overwrite = True)
    # Image shifting
    train_datagen = ImageDataGenerator(width_shift_range=0.05)
    train_datagen.fit(X)
    print("Count of y class labels (should be even)",Counter(np.array(y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RS, stratify=y)
    count_y_train = Counter(y_train)
    count_y_test = Counter(y_test)

    print("Count of y_train class labels (should be even)",count_y_train)
    print("Count of y_test class labels (should be even)",count_y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    mcp_save = ModelCheckpoint("models/rcnn_model_weights.best.hdf5", save_best_only=True, monitor='val_loss', mode='min')
    # Fit model using ImageDataGenerator
    train_model = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                                steps_per_epoch = math.ceil(len(X_train) / batch_size),
                                epochs = EPOCHS1,
                                validation_steps = math.ceil(len(X_test) / batch_size), 
                                verbose = 1,
                                callbacks=[es,tb, mcp_save, rlrl], 
                                validation_data=(train_datagen.flow(X_test,y_test, batch_size = batch_size)),
                                shuffle = True)
    print("Average Accuracy: ", round(np.mean(train_model.history['acc']),4), "% +/- ", round(np.std(train_model.history['acc']),2), "% (Max = ",round(np.max(train_model.history['acc']),4),"%)")
    print("Average Validation Accuracy: ",round(np.mean(train_model.history['val_acc']),4), "% +/- ", round(np.std(train_model.history['val_acc']),2), "% (Max = ",round(np.max(train_model.history['val_acc']),4),"%)")
    print("Average Loss: ", round(np.mean(train_model.history['loss']),4), " +/- ", round(np.std(train_model.history['loss']),2), " (Min = ",round(np.min(train_model.history['loss']),4),")")
    print("Average Validation Loss: ",round(np.mean(train_model.history['val_loss']),4), " +/- ", round(np.std(train_model.history['val_loss']),2), " (Min = ",round(np.min(train_model.history['val_loss']),4),")")
    print("Min Loss: ", round(np.min(train_model.history['loss']),4))
    print("Min Validation Loss: ", round(np.min(train_model.history['val_loss']),4))                                   
    print(model.summary())
    print("rcnn_model Model Output ^^^, model fit complete===========================")
    layer_names = ["Conv2d1","MPool2d1","Conv2d2","MPool2d2","FC128d1","output_layer"]
    if save_accuracy_and_loss_plots == 1:
        print("Plotting Accuracy and Loss Curves." )
        plt.figure(1)
        plt.plot(train_model.history['acc'], label = "Train Acc.",marker='', color='skyblue', linewidth=4)
        plt.plot(train_model.history['val_acc'], label = "Validation Acc.", marker='', color='crimson', linewidth=2)
        plt.title('RCNN_Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(('Train Acc.','Validation Acc.'))

        plt.savefig('rcnn_model_accuracy.png')    
        plt.close()

        plt.figure(2)
        plt.plot(train_model.history['loss'], label = "Train Loss",marker='', color='skyblue', linewidth=4)
        plt.plot(train_model.history['val_loss'], label = "Validation Loss", marker='', color='salmon', linewidth=2)
        plt.title('RCNN_Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(('Train Loss','Validation Loss'))
        plt.savefig('rcnn_model_loss.png')       
        plt.close()
    #train_model output  
    print("=================================================================")    
    print("rcnn_model model output ========================================")
    print("=================================================================")    
    y_predicted_1 = model.predict_classes(X_test)        
    # Print statistics

    print("Count of Predictions from rcnn_model on  test data: \n", Counter(y_predicted_1))
    print('Confusion matrix of test samples(X_test, rcnn_model):\n', np.sum(confusion_matrix(y_predicted_1, y_test),axis=0))
    print('Confusion matrix :\n',confusion_matrix(y_predicted_1, y_test))
    print('Accuracy :', round(get_accuracy(y_predicted_1, y_test),4))
    test_datagen = ImageDataGenerator(width_shift_range=0.05)
    y_predictions_rcnn = model.predict_classes(X_subm)
    y_prob_rcnn = model.predict_proba(X_subm)

    print("rcnn_model count of predictions, submission data: ",Counter(y_predictions_rcnn))
    y_subm_rcnn = Counter(y_predictions_rcnn)
    print("rcnn_model first 10 predictions: ",y_predictions_rcnn[:10])
    if save_predictions == 1:
        L = list(range(5377))
        prediction = pd.DataFrame({'file_id':[20000+c for c in L],
                                    'accent': y_predictions_rcnn,
                                    'probs_0': [elem[0] for elem in y_prob_rcnn],
                                    'probs_1': [elem[1] for elem in y_prob_rcnn],
                                    'probs_2': [elem[2] for elem in y_prob_rcnn]}).to_csv('prediction2.csv',index=False)

    return model, y_subm_rcnn, y_predictions_rcnn, y_prob_rcnn

def no_split_vgg_model(X, y, X_subm, batch_size=64, load_weights = 0): #64
    '''
    Trains 2D convolutional neural network
    :param X_train: Numpy array of mfccs
    :param y_train: Binary matrix based on labels
    :return: Trained model
    '''
    # Get row, column, and class sizes
    rows = X[0].shape[0]
    cols = X[0].shape[1]
    val_rows = X_subm[0].shape[0]
    val_cols = X_subm[0].shape[1]
    num_classes = 3
    # input image dimensions to feed into 2D ConvNet Input layer
    input_shape = (rows, cols, 1)
    
    # Stops training if accuracy does not change at least 0.005 over 10 epochs
    
    rlrl = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=7, verbose=1, min_delta=1e-4, mode='min')
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size, write_graph=True  , write_grads=True,
                    write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                    embeddings_metadata=None)
    model = create_vgg_model(input_shape = input_shape, num_classes=num_classes, load_weights = load_weights)    
    model.save_weights("models/vgg_model_weights.best.hdf5", overwrite = True)
    # Image shifting
    train_datagen = ImageDataGenerator(width_shift_range=0.05)
    train_datagen.fit(X)
    print("Count of y class labels (should be even)",Counter(np.array(y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RS, stratify=y)
    count_y_train = Counter(y_train)
    count_y_test = Counter(y_test)

    print("Count of y_train class labels (should be even)",count_y_train)
    print("Count of y_test class labels (should be even)",count_y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    mcp_save = ModelCheckpoint("models/vgg_model_weights.best.hdf5", save_best_only=True, monitor='val_loss', mode='min')
    # Fit model using ImageDataGenerator
    train_model = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                                steps_per_epoch = math.ceil(len(X_train) / batch_size),
                                epochs = EPOCHS1,
                                validation_steps = math.ceil(len(X_test) / batch_size), 
                                verbose = 1,
                                callbacks=[es,tb, mcp_save, rlrl], 
                                validation_data=(train_datagen.flow(X_test,y_test, batch_size = batch_size)),
                                shuffle = True)
    print("Average Accuracy: ", round(np.mean(train_model.history['acc']),4), "% +/- ", round(np.std(train_model.history['acc']),2), "% (Max = ",round(np.max(train_model.history['acc']),4),"%)")
    print("Average Validation Accuracy: ",round(np.mean(train_model.history['val_acc']),4), "% +/- ", round(np.std(train_model.history['val_acc']),2), "% (Max = ",round(np.max(train_model.history['val_acc']),4),"%)")
    print("Average Loss: ", round(np.mean(train_model.history['loss']),4), " +/- ", round(np.std(train_model.history['loss']),2), " (Min = ",round(np.min(train_model.history['loss']),4),")")
    print("Average Validation Loss: ",round(np.mean(train_model.history['val_loss']),4), " +/- ", round(np.std(train_model.history['val_loss']),2), " (Min = ",round(np.min(train_model.history['val_loss']),4),")")
    print("Min Loss: ", round(np.min(train_model.history['loss']),4))
    print("Min Validation Loss: ", round(np.min(train_model.history['val_loss']),4))                            
    print(model.summary())
    print("vgg_model Model Output ^^^, model fit complete===========================")
    layer_names = ["Conv2d1","MPool2d1","Conv2d2","MPool2d2","Conv2d3","MPool2d3","FC128d1","output_layer"]
    if save_accuracy_and_loss_plots == 1:
        print("Plotting Accuracy and Loss Curves." )
        plt.figure(1)
        plt.plot(train_model.history['acc'], label = "Train Acc.",marker='', color='skyblue', linewidth=4)
        plt.plot(train_model.history['val_acc'], label = "Validation Acc.", marker='', color='crimson', linewidth=2)
        plt.title('VGG_Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(('Train Acc.','Validation Acc.'))

        plt.savefig('train_model_accuracy.png')    
        plt.close()                    

        plt.figure(2)
        plt.plot(train_model.history['loss'], label = "Train Loss",marker='', color='skyblue', linewidth=4)
        plt.plot(train_model.history['val_loss'], label = "Validation Loss", marker='', color='salmon', linewidth=2)
        plt.title('VGG_Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(('Train Loss','Validation Loss'))
        plt.savefig('train_model_loss.png')       
        plt.close()
    #train_model output  
    print("=================================================================")    
    print("vgg_model model output ========================================")
    print("=================================================================")    
    y_predicted_1 = model.predict_classes(X_test)        
    # Print statistics
    
    print("Count of Predictions from vgg_model on  test data: \n", Counter(y_predicted_1))
    print('Confusion matrix of test samples(X_test, vgg_model):\n', np.sum(confusion_matrix(y_predicted_1, y_test),axis=0))
    print('Confusion matrix :\n',confusion_matrix(y_predicted_1, y_test))
    print('Accuracy :', round(get_accuracy(y_predicted_1, y_test),4))
    test_datagen = ImageDataGenerator(width_shift_range=0.05)
    y_predictions_vgg = model.predict_classes(X_subm)
    y_prob_vgg = model.predict_proba(X_subm)

    print("vgg_model count of predictions, submission data: ",Counter(y_predictions_vgg))
    y_subm_vgg = Counter(y_predictions_vgg)
    print("vgg_model first 10 predictions: ",y_predictions_vgg[:10])
    if save_predictions == 1:
        L = list(range(5377))
        prediction = pd.DataFrame({'file_id':[20000+c for c in L],
                                    'accent': y_predictions_vgg,
                                    'probs_0': [elem[0] for elem in y_prob_vgg],
                                    'probs_1': [elem[1] for elem in y_prob_vgg],
                                    'probs_2': [elem[2] for elem in y_prob_vgg]}).to_csv('prediction3.csv',index=False)

    return model, y_subm_vgg, y_predictions_vgg, y_prob_vgg

def cnn_model(X, y, X_subm, batch_size=64, load_weights = 0, n_folds = N_FOLDS):
    # Get row, column, and class sizes
    rows = X[0].shape[0]
    cols = X[0].shape[1]
    val_rows = X_subm[0].shape[0]
    val_cols = X_subm[0].shape[1]
    num_classes = 3
    # input image dimensions to feed into 2D ConvNet Input layer
    input_shape = (rows, cols, 1)

    # Stops training if accuracy does not change at least 0.005 over 10 epochs
    rlrl = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=7, verbose=1, min_delta=1e-4, mode='min')
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=True,
                    write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                    embeddings_metadata=None)
    min_loss_list = []
    groups = np.array(int((len(X)/n_folds))*([1,2,3]))
    kf = GroupKFold(n_splits = n_folds)
    for j, (train_index, test_index) in enumerate(kf.split(X,y,groups)):
        print("Starting on fold ", str(j+1),"/",kf.get_n_splits(X,y,groups)," for cnn model")
        train_datagen = ImageDataGenerator(width_shift_range=0.05)
        train_datagen.fit(X)
        X_train, X_test  = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        classifier = create_cnn_model(input_shape = input_shape, num_classes=num_classes, load_weights = load_weights, string = "")
        mcp_save = ModelCheckpoint("models/cnn_model_weights"+ str(j+1)+ ".best.hdf5", save_best_only=True, monitor='val_loss', mode='min')
        train_model = classifier.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                               steps_per_epoch = math.ceil(len(X_train) / batch_size),
                               epochs = EPOCHS1,
                               validation_steps = math.ceil(len(X_test) / batch_size), 
                               verbose = 1,
                               callbacks=[es,tb, mcp_save, rlrl], 
                               validation_data=(train_datagen.flow(X_test,y_test,batch_size = batch_size)),
                               shuffle = True)
        print("Completed fold ", str(j+1),"/",kf.get_n_splits(X)," for cnn model")
        print("Average Accuracy: ", round(np.mean(train_model.history['acc']),4), "% +/- ", round(np.std(train_model.history['acc']),2), "% (Max = ",round(np.max(train_model.history['acc']),4),"%)")
        print("Average Validation Accuracy: ",round(np.mean(train_model.history['val_acc']),4), "% +/- ", round(np.std(train_model.history['val_acc']),2), "% (Max = ",round(np.max(train_model.history['val_acc']),4),"%)")
        print("Average Loss: ", round(np.mean(train_model.history['loss']),4), " +/- ", round(np.std(train_model.history['loss']),2), " (Min = ",round(np.min(train_model.history['loss']),4),")")
        print("Average Validation Loss: ",round(np.mean(train_model.history['val_loss']),4), " +/- ", round(np.std(train_model.history['val_loss']),2), " (Min = ",round(np.min(train_model.history['val_loss']),4),")")
        print("Min Loss: ", round(np.min(train_model.history['loss']),4))
        print("Min Validation Loss: ", round(np.min(train_model.history['val_loss']),4))
        min_loss_list.append(np.min(train_model.history['val_loss']))
    min_fold = min_loss_list.index(np.min(min_loss_list)) + 1
    print("Fold Number: ", min_fold, " Minimum Valid Loss across all folds: ",round(np.min(min_loss_list),4))

    classifier = create_cnn_model(input_shape = input_shape, num_classes=num_classes, load_weights = 1, string = str(min_fold))
    classifier.save_weights("models/cnn_model_weights.best.hdf5", overwrite = True)

    print(classifier.summary())
    print("CNN Model Output ^^^, model fit complete===========================")
    if save_accuracy_and_loss_plots == 1:
        print("Plotting Accuracy and Loss Curves." )

        plt.figure(1)
        plt.plot(train_model.history['acc'], label = "Train Acc.",marker='', color='skyblue', linewidth=4)
        plt.plot(train_model.history['val_acc'], label = "Validation Acc.", marker='', color='crimson', linewidth=2)
        plt.title('CNN Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(('Train Acc.','Validation Acc.'))
        plt.savefig('cnn_model_accuracy.png')    
        plt.close()                    

        plt.figure(2)
        plt.plot(train_model.history['loss'], label = "Train Loss",marker='', color='skyblue', linewidth=4)
        plt.plot(train_model.history['val_loss'], label = "Validation Loss", marker='', color='salmon', linewidth=2)
        plt.title('CNN Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(('Train Loss','Validation Loss'))
        plt.savefig('cnn_model_loss.png')       
        plt.close()
    #train_model output  
    print("=================================================================")    
    print("cnn_model model output ========================================")
    print("=================================================================")    
    validation_accuracies = []

    for j, (train_index, test_index) in enumerate(kf.split(X,y,groups)):
        print("Starting on fold ", str(j+1),"/",kf.get_n_splits(X,y,groups)," for cnn model")
        train_datagen = ImageDataGenerator(width_shift_range=0.05)
        train_datagen.fit(X)
        X_train, X_test  = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        y_predicted_1 = classifier.predict_classes(X_test)
        # Print statistics
        print("Count of Predictions from cnn_model on Fold "+str(j+1)+" test data: \n", Counter(y_predicted_1))
        print('Confusion matrix of Fold '+str(j+1)+' test samples(X_test, cnn_model):\n', np.sum(confusion_matrix(y_predicted_1, y_test),axis=0))
        print('Confusion matrix (Fold '+str(j+1)+'):\n',confusion_matrix(y_predicted_1, y_test))
        print('Accuracy (Fold '+str(j+1)+'):', round(get_accuracy(y_predicted_1, y_test),4))
        validation_accuracies.append((j+1,round(get_accuracy(y_predicted_1, y_test),4)))
    print("Validation Accuracies (Fold, Validation Accuracy)", validation_accuracies)        


    test_datagen = ImageDataGenerator(width_shift_range=0.05)
    test_datagen.fit(X_subm)

    y_predictions = classifier.predict_classes(X_subm)
    print("cnn_model count of predictions, submission data: ", Counter(y_predictions))

    print("cnn_model first 10 predictions: ",y_predictions[:10])
    if save_predictions == 1:
        L = list(range(5377))
        prediction = pd.DataFrame({'file_id':[20000+c for c in L],
                                    'accent': y_predictions}).to_csv('prediction1.csv',index=False)
    return classifier



def rcnn_model(X, y, X_subm, batch_size=64, load_weights = 0, n_folds = N_FOLDS): #64
    '''
    Trains 2D convolutional neural network
    :param X_train: Numpy array of mfccs
    :param y_train: Binary matrix based on labels
    :return: Trained model
    '''
    # Get row, column, and class sizes
    rows = X[0].shape[0]
    cols = X[0].shape[1]
    val_rows = X_subm[0].shape[0]
    val_cols = X_subm[0].shape[1]
    num_classes = 3
    # input image dimensions to feed into 2D ConvNet Input layer
    input_shape = (rows, cols, 1)
    
    # Stops training if accuracy does not change at least 0.005 over 10 epochs
    
    rlrl = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=7, verbose=1, min_delta=1e-4, mode='min')
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size, write_graph=True  , write_grads=True,
                    write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                    embeddings_metadata=None)

    min_loss_list = []
    groups = np.array(int((len(X)/n_folds))*([1,2,3]))
    kf = GroupKFold(n_splits = n_folds)
    for j, (train_index, test_index) in enumerate(kf.split(X,y,groups)):
        print("Starting on fold ", str(j+1),"/",kf.get_n_splits(X,y,groups)," for rcnn model")
        train_datagen = ImageDataGenerator(width_shift_range=0.05)
        train_datagen.fit(X)
        X_train, X_test  = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        model = create_rcnn_model(input_shape = input_shape, num_classes=num_classes, load_weights = load_weights)
        mcp_save = ModelCheckpoint("models/rcnn_model_weights"+str(j+1)+".best.hdf5", save_best_only=True, monitor='val_loss', mode='min')
        train_model = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                               steps_per_epoch = math.ceil(len(X_train) / batch_size),
                               epochs = EPOCHS2,
                               validation_steps = math.ceil(len(X_test) / batch_size), 
                               verbose = 1,
                               callbacks=[es,tb, mcp_save, rlrl], 
                               validation_data=(train_datagen.flow(X_test,y_test,batch_size = batch_size)),
                               shuffle = True)
        print("Completed fold ", str(j+1),"/",kf.get_n_splits(X)," for rcnn model")
        print("Average Accuracy: ", round(np.mean(train_model.history['acc']),4), "% +/- ", round(np.std(train_model.history['acc']),2), "% (Max = ",round(np.max(train_model.history['acc']),4),"%)")
        print("Average Validation Accuracy: ",round(np.mean(train_model.history['val_acc']),4), "% +/- ", round(np.std(train_model.history['val_acc']),2), "% (Max = ",round(np.max(train_model.history['val_acc']),4),"%)")
        print("Average Loss: ", round(np.mean(train_model.history['loss']),4), " +/- ", round(np.std(train_model.history['loss']),2), " (Min = ",round(np.min(train_model.history['loss']),4),")")
        print("Average Validation Loss: ",round(np.mean(train_model.history['val_loss']),4), " +/- ", round(np.std(train_model.history['val_loss']),2), " (Min = ",round(np.min(train_model.history['val_loss']),4),")")
        print("Min Loss: ", round(np.min(train_model.history['loss']),4))
        print("Min Validation Loss: ", round(np.min(train_model.history['val_loss']),4))
        min_loss_list.append(np.min(train_model.history['val_loss']))
    min_fold = min_loss_list.index(np.min(min_loss_list)) + 1
    print("Fold Number: ", min_fold, " Minimum Valid Loss across all folds: ",round(np.min(min_loss_list),4))
    model = create_rcnn_model(input_shape = input_shape, num_classes=num_classes, load_weights = 1, string = str(min_fold))
    model.save_weights("models/rcnn_model_weights.best.hdf5", overwrite = True)
    print(model.summary())
    print("rcnn_model Model Output ^^^, model fit complete===========================")
    if save_accuracy_and_loss_plots == 1:
        print("Plotting Accuracy and Loss Curves." )
        plt.figure(1)
        plt.plot(train_model.history['acc'], label = "Train Acc.",marker='', color='skyblue', linewidth=4)
        plt.plot(train_model.history['val_acc'], label = "Validation Acc.", marker='', color='crimson', linewidth=2)
        plt.title('RCNN_Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(('Train Acc.','Validation Acc.'))

        plt.savefig('rcnn_model_accuracy.png')    
        plt.close()                    

        plt.figure(2)
        plt.plot(train_model.history['loss'], label = "Train Loss",marker='', color='skyblue', linewidth=4)
        plt.plot(train_model.history['val_loss'], label = "Validation Loss", marker='', color='salmon', linewidth=2)
        plt.title('RCNN_Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(('Train Loss','Validation Loss'))
        plt.savefig('rcnn_model_loss.png')       
        plt.close()
    #train_model output  
    print("=================================================================")    
    print("rcnn_model model output ========================================")
    print("=================================================================")    
    validation_accuracies = []
    for j, (train_index, test_index) in enumerate(kf.split(X,y,groups)):
        print("Starting on fold ", str(j+1),"/",kf.get_n_splits(X,y,groups)," for rcnn model")
        train_datagen = ImageDataGenerator(width_shift_range=0.05)
        train_datagen.fit(X)
        X_train, X_test  = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        y_predicted_1 = model.predict_classes(X_test)        
        # Print statistics
        print("Count of Predictions from rcnn_model on Fold "+str(j+1)+" test data: \n", Counter(y_predicted_1))
        print('Confusion matrix of Fold '+str(j+1)+' test samples(X_test, rcnn_model):\n', np.sum(confusion_matrix(y_predicted_1, y_test),axis=0))
        print('Confusion matrix (Fold '+str(j+1)+'):\n',confusion_matrix(y_predicted_1, y_test))
        print('Accuracy (Fold '+str(j+1)+'):', round(get_accuracy(y_predicted_1, y_test),4))
        validation_accuracies.append((j+1,round(get_accuracy(y_predicted_1, y_test),4)))
    print("Validation Accuracies (Fold, Validation Accuracy)", validation_accuracies)    
    test_datagen = ImageDataGenerator(width_shift_range=0.05)
    test_datagen.fit(X_subm)
    y_predictions = model.predict_classes(X_subm)
    print("rcnn_model count of predictions, submission data: ",Counter(y_predictions))

    print("rcnn_model first 10 predictions: ",y_predictions[:10])
    if save_predictions == 1:
        L = list(range(5377))
        prediction = pd.DataFrame({'file_id':[20000+c for c in L],
                                    'accent': y_predictions}).to_csv('prediction2.csv',index=False)

    return (model)


def vgg_model(X, y, X_subm, batch_size=64, load_weights = 0, n_folds = N_FOLDS): #64
    '''
    Trains 2D convolutional neural network
    :param X_train: Numpy array of mfccs
    :param y_train: Binary matrix based on labels
    :return: Trained model
    '''
    # Get row, column, and class sizes
    rows = X[0].shape[0]
    cols = X[0].shape[1]
    val_rows = X_subm[0].shape[0]
    val_cols = X_subm[0].shape[1]
    num_classes = 3
    # input image dimensions to feed into 2D ConvNet Input layer
    input_shape = (rows, cols, 1)
    
    # Stops training if accuracy does not change at least 0.005 over 10 epochs
    
    rlrl = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=7, verbose=1, min_delta=1e-4, mode='min')
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size, write_graph=True  , write_grads=True,
                    write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                    embeddings_metadata=None)

    min_loss_list = []
    groups = np.array(int((len(X)/n_folds))*([1,2,3]))
    kf = GroupKFold(n_splits = n_folds)
    for j, (train_index, test_index) in enumerate(kf.split(X,y,groups)):
        print("Starting on fold ", str(j+1),"/",kf.get_n_splits(X,y,groups)," for vgg model")
        train_datagen = ImageDataGenerator(width_shift_range=0.05)
        train_datagen.fit(X)
        X_train, X_test  = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        model = create_vgg_model(input_shape = input_shape, num_classes=num_classes, load_weights = load_weights)
        mcp_save = ModelCheckpoint("models/vgg_model_weights"+str(j+1)+".best.hdf5", save_best_only=True, monitor='val_loss', mode='min')
        train_model = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                               steps_per_epoch = math.ceil(len(X_train) / batch_size),
                               epochs = EPOCHS3,
                               validation_steps = math.ceil(len(X_test) / batch_size), 
                               verbose = 1,
                               callbacks=[es,tb, mcp_save, rlrl], 
                               validation_data=(train_datagen.flow(X_test,y_test,batch_size = batch_size)),
                               shuffle = True)
        print("Completed fold ", str(j+1),"/",kf.get_n_splits(X)," for vgg model")
        print("Average Accuracy: ", round(np.mean(train_model.history['acc']),4), "% +/- ", round(np.std(train_model.history['acc']),2), "% (Max = ",round(np.max(train_model.history['acc']),4),"%)")
        print("Average Validation Accuracy: ",round(np.mean(train_model.history['val_acc']),4), "% +/- ", round(np.std(train_model.history['val_acc']),2), "% (Max = ",round(np.max(train_model.history['val_acc']),4),"%)")
        print("Average Loss: ", round(np.mean(train_model.history['loss']),4), " +/- ", round(np.std(train_model.history['loss']),2), " (Min = ",round(np.min(train_model.history['loss']),4),")")
        print("Average Validation Loss: ",round(np.mean(train_model.history['val_loss']),4), " +/- ", round(np.std(train_model.history['val_loss']),2), " (Min = ",round(np.min(train_model.history['val_loss']),4),")")
        print("Min Loss: ", round(np.min(train_model.history['loss']),4))
        print("Min Validation Loss: ", round(np.min(train_model.history['val_loss']),4))
        min_loss_list.append(np.min(train_model.history['val_loss']))

    min_fold = min_loss_list.index(np.min(min_loss_list)) + 1
    print("Fold Number: ", min_fold, " Minimum Valid Loss across all folds: ",round(np.min(min_loss_list),4))

    model = create_vgg_model(input_shape = input_shape, num_classes=num_classes, load_weights = 1, string = str(min_fold))
    model.save_weights("models/vgg_model_weights.best.hdf5", overwrite = True)
    print(model.summary())
    print("vgg_model Model Output ^^^, model fit complete===========================")
    if save_accuracy_and_loss_plots == 1:
        print("Plotting Accuracy and Loss Curves." )
        plt.figure(1)
        plt.plot(train_model.history['acc'], label = "Train Acc.",marker='', color='skyblue', linewidth=4)
        plt.plot(train_model.history['val_acc'], label = "Validation Acc.", marker='', color='crimson', linewidth=2)
        plt.title('VGG_Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(('Train Acc.','Validation Acc.'))

        plt.savefig('vgg_model_accuracy.png')    
        plt.close()                    

        plt.figure(2)
        plt.plot(train_model.history['loss'], label = "Train Loss",marker='', color='skyblue', linewidth=4)
        plt.plot(train_model.history['val_loss'], label = "Validation Loss", marker='', color='salmon', linewidth=2)
        plt.title('VGG_Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(('Train Loss','Validation Loss'))
        plt.savefig('vgg_model_loss.png')       
        plt.close()
    #train_model output  
    print("=================================================================")    
    print("vgg_model model output ========================================")
    print("=================================================================")    
    validation_accuracies = []
    for j, (train_index, test_index) in enumerate(kf.split(X,y,groups)):
        print("Starting on fold ", str(j+1),"/",kf.get_n_splits(X,y,groups)," for vgg model")
        train_datagen = ImageDataGenerator(width_shift_range=0.05)
        train_datagen.fit(X)
        X_train, X_test  = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        y_predicted_1 = model.predict_classes(X_test)        
        # Print statistics
        print("Count of Predictions from vgg_model on Fold "+str(j+1)+" test data: \n", Counter(y_predicted_1))
        print('Confusion matrix of Fold '+str(j+1)+' test samples(X_test, vgg_model):\n', np.sum(confusion_matrix(y_predicted_1, y_test),axis=0))
        print('Confusion matrix (Fold '+str(j+1)+'):\n',confusion_matrix(y_predicted_1, y_test))
        print('Accuracy (Fold '+str(j+1)+'):', round(get_accuracy(y_predicted_1, y_test),4))
        validation_accuracies.append((j+1,round(get_accuracy(y_predicted_1, y_test),4)))
    print("Validation Accuracies (Fold, Validation Accuracy)", validation_accuracies)    
    test_datagen = ImageDataGenerator(width_shift_range=0.05)
    test_datagen.fit(X_subm)
    y_predictions = model.predict_classes(X_subm)
    print("vgg_model count of predictions, submission data: ",Counter(y_predictions))

    print("vgg_model first 10 predictions: ",y_predictions[:10])
    if save_predictions == 1:
        L = list(range(5377))
        prediction = pd.DataFrame({'file_id':[20000+c for c in L],
                                    'accent': y_predictions}).to_csv('prediction3.csv',index=False)

    return (model)


def save_model(model, model_filename):
    '''
    Save model to file
    :param model: Trained model to be saved
    :param model_filename: Filename
    :return: None
    '''
    model.save('./models/{}.h5'.format(model_filename))  # creates a HDF5 file 'my_model.h5'

def plot_cnn_layer_visualizations(outputs,cnn_layer_names,filename):
    #plotting the outputs
    fig,ax = plt.subplots(nrows=6,ncols=6,figsize=(36,36))    
    for i in range(4):
        for z in range(6):
            ax[i][z].imshow(outputs[i][0,:,:,z])
            ax[i][z].set_title(cnn_layer_names[i])
            ax[i][z].set_xticks([])
            ax[i][z].set_yticks([])
    for i in range(4,6):
        for z in range(6):
            ax[i][z].imshow(outputs[i][:,:])
            ax[i][z].set_title(cnn_layer_names[i])
            ax[i][z].set_xticks([])
            ax[i][z].set_yticks([])
    plt.savefig('cnn_layerwise_output'+str(filename)+'.jpg')

def plot_rcnn_layer_visualizations(outputs, rcnn_layer_names,filename):
    #plotting the outputs
    fig,ax = plt.subplots(nrows=6,ncols=6,figsize=(36,36))
    for i in range(4):
        for z in range(6):
            ax[i][z].imshow(outputs[i][0,:,:,z])
            ax[i][z].set_title(rcnn_layer_names[i])
            ax[i][z].set_xticks([])
            ax[i][z].set_yticks([])
    for i in range(4,6):
        for z in range(6):
            ax[i][z].imshow(outputs[i][:,:])
            ax[i][z].set_title(rcnn_layer_names[i])
            ax[i][z].set_xticks([])
            ax[i][z].set_yticks([])
    plt.savefig('rcnn_layerwise_output'+str(filename)+'.jpg')

def plot_vgg_layer_visualizations(outputs, rcnn_layer_names,filename):      
        #plotting the outputs
    fig,ax = plt.subplots(nrows=8,ncols=8,figsize=(64, 64))

    for i in range(6):
        for z in range(8):
            ax[i][z].imshow(outputs[i][0,:,:,z])
            ax[i][z].set_title(vgg_layer_names[i])
            ax[i][z].set_xticks([])
            ax[i][z].set_yticks([])
    for i in range(6,8):
        for z in range(8):
            ax[i][z].imshow(outputs[i][:,:])
            ax[i][z].set_title(vgg_layer_names[i])
            ax[i][z].set_xticks([])
            ax[i][z].set_yticks([])

    plt.savefig('vgg_layerwise_output'+str(filename)+'.jpg')  
############################################################




#######################################

if __name__ == '__main__':
    '''
        Console command example:
        '''

    train_path = 'train/'
    test_path = 'test/'
    contents = os.listdir(train_path)
    test_contents = os.listdir(test_path)
    file_name = 'train_labels.csv'

    cnn_model_filename = 'model1cnn'
    model_filename = 'model2rcnn'
    vgg_model_filename = 'model3vgg'


    # Load metadata
    df = pd.read_csv(file_name)

    batch = [] ## Empty list of image list
    labels = []  ## Empty image labels list

    for ii, file in enumerate(contents, 10000):  ## Enumerate over each image list
        string = str(file[:-4])
        img = io.imread(train_path+file,as_gray=True)  ## each the images from the folder. We are passing the file path + name in "imread"
        #img = normalize_mfcc(img)
        # img = remove_silence(img)
        if plot_layer_visualizations_flag == 1:
            if ii == 10000:
                image = img.reshape(128, 173, 1)
                io.imsave('img10000.png',img)
            if ii == 10002:
                image1 = img.reshape(128, 173, 1)
            if ii == 10003:
                image2 = img.reshape(128, 173, 1)            
        accent = list(df[df['file_id']==int(string)]['accent'])[0]
        batch.append(img.reshape(128, 173, 1))  ## reshaping images and append in one file list
        labels.append(accent) ## appending the labels of each image
    print('batch length:', len(batch))
    print('labels length:', len(labels))

    test_batch = [] ## Empty list of image list


    for ii, file in enumerate(test_contents, 20000):  ## Enumerate over each image list
        string = str(file[:-4])
        img = io.imread(test_path+file,as_gray=True)  
        #img = normalize_mfcc(img)
        ## each the images from the folder. We are passing the file path + name in "imread"
        # img = remove_silence(img)
        if plot_layer_visualizations_flag == 1:
            if ii == 20000:
                io.imsave('img20000.png',img)
        
        test_batch.append(img.reshape(128, 173, 1))  ## reshaping images and append in one file list
    print('test_batch length:', len(test_batch))


    X_subm = np.asarray(test_batch)
    X_train, X_val, y_train, y_val = train_test_split(batch, labels, test_size=VAL_SPLIT, random_state=0, stratify = labels)

    if train_models == 1:
        # Train model
        print("Now beginning fitting for CNN model ===========================")
        cnn_model,  y_subm_cnn_count, y_subm_cnn, y_prob_cnn = no_split_cnn_model(np.array(X_train), np.array(y_train), X_subm, 
                        batch_size= BATCH_SIZE1, load_weights=LOAD_WEIGHTS_CNN)
        print("CNN model fitting completed." )
        if plot_layer_visualizations_flag == 1:
            cnn_layer_names = ["Conv2d1","MPool2d1","Conv2d2","MPool2d2","FC128d1","output_layer"]
            outputs = []
            outputs1 = []
            outputs2 = []
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
            image2 = image2.reshape((1, image2.shape[0], image2.shape[1], image2.shape[2]))

            #extracting the output and appending to outputs
            for layer_name in cnn_layer_names:
                intermediate_layer_model = Model(inputs=cnn_model.input,outputs=cnn_model.get_layer(layer_name).output)
                intermediate_output = intermediate_layer_model.predict(image)
                outputs.append(intermediate_output)
                intermediate_output1 = intermediate_layer_model.predict(image1)
                outputs1.append(intermediate_output1)
                intermediate_output2 = intermediate_layer_model.predict(image2)
                outputs2.append(intermediate_output2)
            #plotting the outputs
            print(outputs[4].shape)
            plot_cnn_layer_visualizations(outputs, cnn_layer_names,10000)
            plot_cnn_layer_visualizations(outputs1, cnn_layer_names,10002)
            plot_cnn_layer_visualizations(outputs2, cnn_layer_names,10003)
        # Save model
        save_model(cnn_model, cnn_model_filename)
        print("Now beginning fitting for rcnn_model===========================")
        rcnn_model, y_subm_rcnn_count, y_subm_rcnn, y_prob_rcnn  = no_split_rcnn_model(np.array(X_train), np.array(y_train), X_subm, 
                                batch_size= BATCH_SIZE2, load_weights=LOAD_WEIGHTS_RCNN)
        print("rcnn_model model fitting completed." )
        if plot_layer_visualizations_flag == 1:
            rcnn_layer_names = ["Conv2d1","MPool2d1","Conv2d2","MPool2d2","FC128d1","output_layer"]
            outputs = []
            outputs1 = []
            outputs2 = []

            #extracting the output and appending to outputs
            for layer_name in rcnn_layer_names:
                intermediate_layer_model = Model(inputs=rcnn_model.input,outputs=rcnn_model.get_layer(layer_name).output)
                intermediate_output = intermediate_layer_model.predict(image)
                outputs.append(intermediate_output)
                intermediate_output1 = intermediate_layer_model.predict(image1)
                outputs1.append(intermediate_output1)
                intermediate_output2 = intermediate_layer_model.predict(image2)
                outputs2.append(intermediate_output2)

            plot_rcnn_layer_visualizations(outputs, rcnn_layer_names,10000)
            plot_rcnn_layer_visualizations(outputs1, rcnn_layer_names,10002)
            plot_rcnn_layer_visualizations(outputs2, rcnn_layer_names,10003)       
        # Save model
        save_model(rcnn_model, model_filename)
        
        print("Now beginning fitting for vgg_model===========================")
        vgg_model,  y_subm_vgg_count, y_subm_vgg, y_prob_vgg  = no_split_vgg_model(np.array(X_train), np.array(y_train), X_subm, 
                                batch_size= BATCH_SIZE3, load_weights=LOAD_WEIGHTS_VGG)
        print("vgg_model model fitting completed." )
        if plot_layer_visualizations_flag == 1:
            vgg_layer_names = ["Conv2d1","MPool2d1","Conv2d2","MPool2d2","Conv2d3","MPool2d3","FC128d1","output_layer"]
            outputs = []
            outputs1 = []
            outputs2 = []
            #extracting the output and appending to outputs
            for layer_name in vgg_layer_names:
                intermediate_layer_model = Model(inputs=vgg_model.input,outputs=vgg_model.get_layer(layer_name).output)
                intermediate_output = intermediate_layer_model.predict(image)
                outputs.append(intermediate_output)
                intermediate_output1 = intermediate_layer_model.predict(image1)
                outputs1.append(intermediate_output1)
                intermediate_output2 = intermediate_layer_model.predict(image2)
                outputs2.append(intermediate_output2)

            plot_vgg_layer_visualizations(outputs, vgg_layer_names,10000)
            plot_vgg_layer_visualizations(outputs1, vgg_layer_names,10002)
            plot_vgg_layer_visualizations(outputs2, vgg_layer_names,10003)    

        # Save model
        save_model(vgg_model, vgg_model_filename)
    else:
        rows = (np.array(X_train))[0].shape[0]
        cols = (np.array(X_train))[0].shape[1]
        num_classes = 3
        # input image dimensions to feed into 2D ConvNet Input layer
        input_shape = (rows, cols, 1)
        model = create_cnn_model(input_shape, num_classes, load_weights=1)
        rcnn_model = create_rcnn_model(input_shape, num_classes, load_weights=1)
        vgg_model = create_vgg_model(input_shape, num_classes, load_weights=1)
#     labels = to_categorical(labels)   




    
    # # Train test split
    # X_train, X_test, y_train, y_test = train_test_split(batch, labels, test_size=0.40, random_state=1234)
    # # Get statistics
    print("=====================================================================")
    print("Validation CNN_Model (20 percent remaing Data) ======================")
    print("=====================================================================")

    y_predicted_1 = cnn_model.predict_classes(np.array(X_val))
    y = to_categorical(np.array(y_val))
    # Print statistics
    print("Count of Predictions from cnn_model on validation data: \n", Counter(y_predicted_1))
    print('Confusion matrix of total samples(X_all, cnn_model):\n', np.sum(confusion_matrix(y_predicted_1, y),axis=0))
    print('Confusion matrix:\n',confusion_matrix(y_predicted_1, y))
    print('Accuracy:', get_accuracy(y_predicted_1, y))
    print("y_subm_cnn counter: ", y_subm_cnn_count)
# # Get statistics

    print("=====================================================================")
    print("Validation rcnn_model (20 percent remaing Data) ====================")
    print("=====================================================================")
    y_predicted_2 = rcnn_model.predict_classes(np.array(X_val))
    y2 = to_categorical(np.array(y_val))
    # Print statistics
    print("Count of Predictions from rcnn_model on validation data: \n", Counter(y_predicted_2))
    print('Confusion matrix of total samples(X_all, rcnn_model):\n', np.sum(confusion_matrix(y_predicted_2, y2),axis=0))
    print('Confusion matrix:\n',confusion_matrix(y_predicted_2, y2))
    print('Accuracy:', get_accuracy(y_predicted_2, y2))
    print("y_subm_rcnn counter: ", y_subm_rcnn_count)
# # Get statistics

    print("=====================================================================")
    print("Validation vgg_model (20 percent remaing Data) ====================")
    print("=====================================================================")
    y_predicted_3 = vgg_model.predict_classes(np.array(X_val))
    y3 = to_categorical(np.array(y_val))
    # Print statistics
    print("Count of Predictions from vgg_model on validation data: \n", Counter(y_predicted_3))
    print('Confusion matrix of total samples(X_all, vgg_model):\n', np.sum(confusion_matrix(y_predicted_3, y3),axis=0))
    print('Confusion matrix:\n',confusion_matrix(y_predicted_3, y3))
    print('Accuracy:', get_accuracy(y_predicted_3, y3))
    print("y_subm_vgg counter: ", y_subm_vgg_count)
    if save_predictions == 1:
        L = list(range(5377))
        prediction = pd.DataFrame({'file_id':[20000+c for c in L],
                                    'accent_cnn': y_subm_cnn,
                                    'accent_rcnn': y_subm_rcnn,
                                    'accent_vgg': y_subm_vgg,
                                    'cnn_probs_0': [elem[0] for elem in y_prob_cnn],
                                    'cnn_probs_1': [elem[1] for elem in y_prob_cnn],
                                    'cnn_probs_2': [elem[2] for elem in y_prob_cnn],
                                    'rcnn_probs_0': [elem[0] for elem in y_prob_rcnn],
                                    'rcnn_probs_1': [elem[1] for elem in y_prob_rcnn],
                                    'rcnn_probs_2': [elem[2] for elem in y_prob_rcnn],                                
                                    'vgg_probs_0': [elem[0] for elem in y_prob_vgg],
                                    'vgg_probs_1': [elem[1] for elem in y_prob_vgg],
                                    'vgg_probs_2': [elem[2] for elem in y_prob_vgg]}).to_csv('predictioncomb.csv',index=False)
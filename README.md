# DAT264x: Identifying Accents in Spectrograms of Speech 
https://datasciencecapstone.org/competitions/16/identifying-accents-speech/
### About
Your goal is to predict the accent of the speaker from spectrograms of speech samples. A spectrogram is a visual representation of the various frequencies of sound as they vary with time. These spectrograms were generated from audio samples in the Mozilla Common Voice dataset. Each speech clip was sampled at 22,050 Hz, and contains an accent from one of the following three countries: Canada, India, and England. For more information on spectrograms, see the home page.

### Dataset
For each observation, you are given a spectrogram of speech audio. The files are named with the convention {file_id}.png. The file_id in the filename matches the file_id column in train_labels.csv for the training data and in submission_format.csv for the test data.

The spectrograms have been scaled so that they are 128x173. Each image only has one channel so if it is properly loaded, the shape should be (128, 173). Using scikit-image, loading a file looks like:

from skimage import io
import matplotlib.pyplot as plt

my_image = io.imread('10000.png', as_gray=True)

# look at the image
plt.imshow(my_image)

Target Variable
The accent labels correspond to the following countries:

0: Canada
1: India
2: England

Submission Format
The format for the submission file is CSV with a header row (file_id, accent). Each row contains a file ID (an integer) and the accent value (an integer) separated by a comma. The file ID corresponds to the filename of the images in the test/ folder. The accent value is the target label, one of {0, 1, 2}. Note that your accent labels must be integers.

### My Models

The MFCC was fed into a three models: 
• 2-Dimensional Convolutional Neural Network (CNN) 
    (73.2% Train Accuracy, 71.7% Test Accuracy) 
• 2-D RCNN 
    (74.9% Train Accuracy, 72.49% Test Accuracy) 
• VGG-ish Neural Network 
    (75.6% Train Accuracy, 72.35% Test Accuracy) 
    
to predict the native language class.

### Dependencies
• [Python 3.x](https://www.python.org/download/releases/python-374/)

• [Keras](https://keras.io/)

• [Numpy](http://www.numpy.org/)

• [Scikit_Image](https://scikit-image.org/)

• [Pandas](https://pandas.pydata.org/index.html)

• [Scikit-learn](http://scikit-learn.org/stable/)

• [Librosa](http://librosa.github.io/librosa/)

• [Pillow](https://pypi.org/project/Pillow/)

• [Skimage](https://scikit-image.org/docs/dev/api/skimage.html)






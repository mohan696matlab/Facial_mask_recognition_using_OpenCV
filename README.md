# COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning
Facial recognition has become more widespread and accurate in recent years, as an artificial intelligence technology called deep learning made computers much better at interpreting images. Governments and private companies use facial recognition to identify people at workplaces, schools, and airports, among other places, although some algorithms perform less well on women and people with darker skin tones. Now the facial-recognition industry is trying to adapt to a world where many people keep their faces covered to avoid spreading disease.

## Steps to be followed
* train_mask_detector  - Accepts our input dataset and fine-tunes VGG-16 upon it to create our mask_detector model. A training history plot.png containing accuracy/loss curves is also produced
* detect_mask_image - Performs face mask detection in static images
* detect_mask_video- Using your webcam, this script applies face mask detection to every frame in the stream

## Dataset
This dataset consists of 1,376 images belonging to two classes:
with_mask: 690 images
without_mask: 686 images
Our goal is to train a custom deep learning model to detect whether a person is or is not wearing a mask.

`import cv2
import keras
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf 
import PIL

import os
from pathlib import Path
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model`

## Convolutiona neural network training
1. Load MobileNet with pre-trained ImageNet weights, leaving off head of network 
2. Construct a new FC head, and append it to the base in place of the old head 
3. Freeze the base layers of the network . The weights of these base layers will not be updated during the process of backpropagation, whereas the head layer weights will be tuned.




# COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning
Facial recognition has become more widespread and accurate in recent years, as an artificial intelligence technology called deep learning made computers much better at interpreting images. Governments and private companies use facial recognition to identify people at workplaces, schools, and airports, among other places, although some algorithms perform less well on women and people with darker skin tones. Now the facial-recognition industry is trying to adapt to a world where many people keep their faces covered to avoid spreading disease.

## Steps to be followed
* train_mask_detector  - Accepts our input dataset and fine-tunes VGG-16 upon it to create our mask_detector model. A training history plot.png containing accuracy/loss curves is also produced
* detect_mask_image - Performs face mask detection in static images
* detect_mask_video- Using your webcam, this script applies face mask detection to every frame in the stream

## Dataset
This dataset consists of 1,376 images belonging to two classes:
* with_mask: 690 images
* without_mask: 686 images
Our goal is to train a custom deep learning model to detect whether a person is or is not wearing a mask.


## Convolutiona neural network training
1. Load MobileNet with pre-trained ImageNet weights, leaving off head of network. 

`vgg = VGG16(input_shape=[100,100] +[3], weights='imagenet', include_top=False)`


2. Construct a new FC head, and append it to the base in place of the old head 

`x = Flatten()(vgg.output) `

`prediction = Dense(1, activation='sigmoid')(x)`

`model = Model(inputs=vgg.input, outputs=prediction)`




3. Freeze the base layers of the network . The weights of these base layers will not be updated during the process of backpropagation, whereas the head layer weights will be tuned.

`for layer in vgg.layers:`
    `layer.trainable = False`




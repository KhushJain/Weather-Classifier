!pip install --upgrade tensorflow

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os

class TransferLearning:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		#model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		base_model = VGG16(input_shape=inputShape, include_top=False, weights="imagenet")
		base_model.trainable = False
		x = Flatten()(base_model.output)
		x = Dense(512,activation = "relu")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.2)(x)
		x = Dense(512,activation = "relu")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.2)(x)
		prediction_layer = Dense(5, activation='softmax')(x)
		
		model = Model(inputs=base_model.input, outputs=prediction_layer)

		# return the constructed network architecture
		return model

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

#Unrar dataset
!unrar x "/content/drive/My Drive/datasets/weather.rar" "/content/"

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 100
INIT_LR = 1e-3
BS = 8
IMAGE_DIMS = (256, 256, 3)

# initialize the data and labels
data = []
labels = []

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("dataset")))
random.seed(42)
random.shuffle(imagePaths)
count = 0
errorPath = []
# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	try:

		print(imagePath)
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
		image = img_to_array(image)
		data.append(image)
	
		# extract the class label from the image path and update the
		# labels list
		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)
  
	except:

		count += 1
		errorPath.append(imagePath)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[LABELS]", labels)
print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))

# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.4, horizontal_flip=True, fill_mode="nearest")

#ModelCheckpoint
checkpoint = ModelCheckpoint('/content/model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# initialize the model
print("[INFO] compiling model...")
model = TransferLearning.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(lb.classes_))
#opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer='Adam',
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1, callbacks=callbacks_list)

# evaluate
result = model.evaluate(testX, testY, batch_size=128)
print(result)

# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open("/content/lb.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()
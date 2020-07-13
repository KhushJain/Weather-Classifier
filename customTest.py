# Testing on a custom image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pickle

# load the model
model = load_model('/content/model.h5')
f = open("/content/lb.pickle", "rb")
lb = pickle.load(f)
f.close()

# load the image
image = cv2.imread(input('Give the complete image path: '))
 
# pre-process the image for classification
image = cv2.resize(image, (256, 256))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print("[INFO] classifying image...")
proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]
print("Output: ", label, "Probability: ", proba[idx] * 100)
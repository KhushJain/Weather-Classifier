from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2

model = load_model("./model/model.h5")
lb = pickle.loads(open('./model/lb.pickle', "rb").read())

def index(request):
    if request.method == 'POST':

        fileObj = request.FILES["filePath"]
        fs=FileSystemStorage()
        filePathName=fs.save(fileObj.name,fileObj)
        filePathName=fs.url(filePathName)
        testimage='.'+filePathName

        try:
            image = cv2.imread(testimage)
            image = cv2.resize(image, (256, 256))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            label = lb.classes_[np.argmax(model.predict(image)[0])]

            context = {
                'filePathName': filePathName,
                'label': label
            }
        except:
            context = {
                'label': 'Image is corrupted'
            }
        return render(request, 'index.html', context)
    else:
        return render(request, 'index.html')
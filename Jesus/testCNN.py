from keras.preprocessing.image import img_to_array
from keras.models import load_model # Used to load the serialized CNN
import numpy as np
import argparse
import imutils
import os
from imutils import paths
import cv2
import csv, operator

# HeightxWidth of the test images
height = 64
width = 64

# Directory where the test images are
directory = "C:\\Users\\Yus\\Desktop\\DeepLearningImageRecognition\\Test"

image_paths = sorted(list(paths.list_images(directory)))
np.random.seed(77189383)
np.random.shuffle(image_paths)
image_paths.sort( key = lambda image: int(image.split("img")[1].split(".")[0]))
data = []
print("[INFO] Loading CNN...")
model = load_model("modelo_8") # Load the model

print("[INFO] Loading test images...")
for image_path in image_paths:
	# load the image, pre-process it, and store it in the data list
	if(image_path[-4:]==".jpg" or image_path[-4:]==".png" or image_path[-4:]==".JPG"):
		image = cv2.imread(image_path)
		orig = image.copy()

		# pre-process the image for classification.
		image = cv2.resize(image, (height, width))
		image = image.astype("float") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis = 0)

		# classify the input image
		(phone, pistol) = model.predict(image)[0]

		# build the label
		label = "Pistol" if pistol > phone else "Phone"
		if label=="Pistol":
			data.append((image_path.split(os.path.sep)[-1] , 0))
		else:
			data.append((image_path.split(os.path.sep)[-1] , 1))

csvsalida = open('modelo_8.csv', 'w', newline='')
salida = csv.writer(csvsalida)
salida.writerow(['Id', 'Ground_Truth'])
salida.writerows(data)
del salida
csvsalida.close()

print("[INFO] Testing finished")
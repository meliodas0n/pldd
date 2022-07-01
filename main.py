import os
os.add_dll_directory(r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin')
os.add_dll_directory(r'C:/CUDNN/cuda/bin')
from tkinter import Tk, Button, CENTER, mainloop
from tkinter import messagebox
from tkinter.filedialog import askopenfilenames
import cv2
import tensorflow as tf
import numpy as np


win = Tk()
win.geometry('1000x800')
MODEL_PATH = r'saved_model2/my_model'
CLASS_NAMES = ['Alstonia_Scholaris___diseased',
 'Alstonia_Scholaris___healthy',
 'Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Arjun___diseased',
 'Arjun___healthy',
 'Background_without_leaves',
 'Bael___diseased',
 'Basil___healthy',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Chinar___diseased',
 'Chinar___healthy',
 'Coffee',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Gauva___diseased',
 'Gauva___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Jamun___diseased',
 'Jamun___healthy',
 'Jatropha___diseased',
 'Jatropha___healthy',
 'Lemon___diseased',
 'Lemon___healthy',
 'Mango___diseased',
 'Mango___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Pomegranate___diseased',
 'Pomogranate___healthy',
 'Pongamia_Pinnata___diseased',
 'Pongamia_Pinnata___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
IMG_HEIGHT = 160
IMG_WIDTH = 160



def take_picture():
	cam = cv2.VideoCapture(0)
	while True:
		ret, frame = cam.read()
		if not ret:
			print('failed to grab frame')
			break
		cv2.imshow('tst', frame)

		k = cv2.waitKey(1)
		if k % 256 == 27:
			print("Escape Hit, closing---")
			break
		elif k % 256 == 32:
			img_name = f"test_image.jpg"
			cv2.imwrite(img_name, frame)
	cam.release()
	cv2.destroyAllWindows()

img_path = r"E:/New folder/test_image.jpg"


def predict_picture():
	model = tf.keras.models.load_model(MODEL_PATH)
	img = tf.keras.utils.load_img(img_path, target_size = (IMG_HEIGHT, IMG_WIDTH))
	img_array = tf.keras.utils.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0)
	predictions = model.predict(img_array)
	score = tf.nn.softmax(predictions[0])
	prediction_text = f"This image most likely belongs to {CLASS_NAMES[np.argmax(score)]} with a {100.00 * np.max(score)} percent confidence."
	messagebox.showinfo("MODEL PREDICTION", prediction_text)


def predict_picture_file(img_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    img = tf.keras.utils.load_img(img_path, target_size = (IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    prediction_text = f"This image most likely belongs to {CLASS_NAMES[np.argmax(score)]} with a {100.00 * np.max(score)} percent confidence."
    messagebox.showinfo("MODEL PREDICTION", prediction_text)


def open_file():
    filenames = askopenfilenames()
    for i in filenames:
        predict_picture_file(i)


b = Button(win, bg = 'black', fg = 'white', height = 5, width = 20, text = 'CAM', anchor = CENTER, command = take_picture)
b.pack()
predict_button = Button(win, bg = 'white', fg = 'black', height = 5, width = 20, text = 'PREDICT', anchor = CENTER, command = predict_picture)
predict_button.pack()
file_open = Button(win, bg = 'white', fg = 'black', height = 5, width = 30, text = 'OPEN FILE & PREDICT', anchor = CENTER, command = open_file)
file_open.pack()

mainloop()
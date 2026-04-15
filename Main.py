from tkinter import *
import tkinter
from tkinter import ttk
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob
import os
import cv2
from keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose, Input
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import imutils
 
from keras.models import load_model
from PIL import Image
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
 
main = tkinter.Tk()
main.title("Intelligent Video Surveillance Using Deep Learning")
main.geometry("1300x1200")
 
global filename
global model
model = None          # FIX 3: initialise to None so the guard in abnormalDetection works
images = []           # always starts as a plain list
 
 
def readImages(path):
    img = load_img(path)
    img = img_to_array(img)
    img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_AREA)
    gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    images.append(gray)
 
 
def upload():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n")
 
 
def datasetPreprocess():
    global filename, images
    # FIX 1: reset to a plain list regardless of current type (list or ndarray)
    images = []
    text.delete('1.0', END)
    img_list = os.listdir(filename)
    for img in img_list:
        print("Dataset/" + img)
        readImages("Dataset/" + img)
    images = np.array(images)
    testImage = images[0]
    height, width, color = images.shape
    images.resize(width, color, height)
    images = (images - images.mean()) / (images.std())
    images = np.clip(images, 0, 1)
    text.insert(END, "Total images found in dataset: " + str(images.shape[0]))
    cv2.imshow("Process Images", testImage / 255)
    cv2.waitKey(0)
 
 
def meanLoss(image1, image2):
    difference = image1 - image2
    a, b, c, d, e = difference.shape
    n_samples = a * b * c * d * e
    sq_difference = difference**2
    Sum = sq_difference.sum()
    distance = np.sqrt(Sum)
    mean_distance = distance / n_samples
    return mean_distance
 
 
def buildModel():
    """Build the STAE model using the Functional API so input_shape is
    declared via a proper Input layer — compatible with Keras 3.x."""
    inputs = Input(shape=(227, 227, 10, 1))          # FIX 2: explicit Input layer
    x = Conv3D(filters=128, kernel_size=(11, 11, 1),
               strides=(4, 4, 1), padding='valid', activation='tanh')(inputs)
    x = Conv3D(filters=64, kernel_size=(5, 5, 1),
               strides=(2, 2, 1), padding='valid', activation='tanh')(x)
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same',
                   dropout=0.4, recurrent_dropout=0.3, return_sequences=True)(x)
    x = ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=1, padding='same',
                   dropout=0.3, return_sequences=True)(x)
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same',
                   dropout=0.5, return_sequences=True)(x)
    x = Conv3DTranspose(filters=128, kernel_size=(5, 5, 1),
                        strides=(2, 2, 1), padding='valid', activation='tanh')(x)
    outputs = Conv3DTranspose(filters=1, kernel_size=(11, 11, 1),
                              strides=(4, 4, 1), padding='valid', activation='tanh')(x)
    stae_model = Model(inputs, outputs)
    stae_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return stae_model
 
 
def trainModel():
    global model
    text.delete('1.0', END)
    if os.path.exists('model/survey_model.h5'):
        try:
            model = load_model("model/survey_model.h5")
            text.insert(END, "Loaded existing model from model/survey_model.h5\n")
        except Exception as e:
            # FIX 2: if the saved h5 is incompatible with current Keras,
            # rebuild the architecture and load only the weights.
            text.insert(END, f"Full model load failed ({e}).\n"
                             "Rebuilding architecture and loading weights only…\n")
            stae_model = buildModel()
            stae_model.load_weights("model/survey_model.h5")
            model = stae_model
            text.insert(END, "Weights loaded successfully.\n")
    else:
        stae_model = buildModel()
        frames = images.shape[2]
        frames = frames - frames % 10
        training_data = images[:, :, :frames]
        training_data = training_data.reshape(-1, 227, 227, 10)
        training_data = np.expand_dims(training_data, axis=4)
        target_data = training_data.copy()
        os.makedirs("model", exist_ok=True)
        callback_save = ModelCheckpoint("model/survey_model.h5",
                                        monitor="mean_squared_error",
                                        save_best_only=True)
        callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        stae_model.fit(training_data, target_data, batch_size=1, epochs=5,
                       callbacks=[callback_save, callback_early_stopping])
        stae_model.save("model/survey_model.h5")
        model = stae_model
    text.insert(END, "Auto Encoder STAE model ready.\n")
 
 
def abnormalDetection():
    global model
    text.delete('1.0', END)
 
    # FIX 3: guard against model not being loaded yet
    if model is None:
        text.insert(END, "ERROR: Please train or load the model first "
                         "(click 'Train Spatial Temporal AutoEncoder Model').\n")
        return
 
    filename = filedialog.askopenfilename(initialdir="testVideos")
    if not filename:
        return
    cap = cv2.VideoCapture(filename)
    print(cap.isOpened())
    while cap.isOpened():
        imagedump = []
        ret, frame = cap.read()
        for i in range(10):
            ret, frame = cap.read()
            if frame is not None:
                image = imutils.resize(frame, width=700, height=600)
                frame = cv2.resize(frame, (227, 227), interpolation=cv2.INTER_AREA)
                gray = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]
                gray = (gray - gray.mean()) / gray.std()
                gray = np.clip(gray, 0, 1)
                imagedump.append(gray)
        if len(imagedump) < 10:          # not enough frames left → stop
            break
        imagedump = np.array(imagedump)
        imagedump.resize(227, 227, 10)
        imagedump = np.expand_dims(imagedump, axis=0)
        imagedump = np.expand_dims(imagedump, axis=4)
        output = model.predict(imagedump)
        loss = meanLoss(imagedump, output)
        if frame is None:
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        print(str(loss))
        if loss > 0.00068:
            print('Abnormal Event Detected')
            cv2.putText(image, "Abnormal Event", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        else:
            cv2.putText(image, "Normal Event", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        cv2.imshow("video", image)
    cap.release()
    cv2.destroyAllWindows()
 
 
def close():
    global main
    main.destroy()
 
 
font = ('times', 16, 'bold')
title = Label(main, text='Intelligent Video Surveillance Using Deep Learning')
title.config(bg='LightGoldenrod1', fg='medium orchid')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)
 
font1 = ('times', 12, 'bold')
text = Text(main, height=30, width=100)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400, y=100)
text.config(font=font1)
 
uploadButton = Button(main, text="Upload Video Frames Dataset", command=upload)
uploadButton.place(x=50, y=100)
uploadButton.config(font=font1)
 
processButton = Button(main, text="Dataset Preprocessing", command=datasetPreprocess)
processButton.place(x=50, y=150)
processButton.config(font=font1)
 
trainButton = Button(main, text="Train Spatial Temporal AutoEncoder Model", command=trainModel)
trainButton.place(x=50, y=200)
trainButton.config(font=font1)
 
testButton = Button(main, text="Test Video Surveillance", command=abnormalDetection)
testButton.place(x=50, y=250)
testButton.config(font=font1)
 
exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50, y=300)
exitButton.config(font=font1)
 
main.config(bg='OliveDrab2')
main.mainloop()
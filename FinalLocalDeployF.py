from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from keras.preprocessing import image
import numpy as np
import pandas as pd
import cv2
import math
from glob import glob
from scipy import stats as s
from keras.models import model_from_json
from PIL import Image
from PIL import ImageFilter

app = Flask(__name__)


@app.route('/')
def upload_file():
    return render_template('Submit1.html')


@app.route('/uploader', methods=['POST'])
def uploader_file():
    if request.method == 'POST':
        if request.files:
            fop1 = request.files["file"]
            print(fop1)
            dhgy = secure_filename(fop1.filename)
            fop1.save(dhgy)
            print("vedio saved")

            # creating the tags
            train = pd.read_csv('D:\\Python\\Pycharm\\PycharmProjects\\IPFinal\\train_new4.csv')
            y = train['class']
            y = pd.get_dummies(y)
            print("y created")

            # creating two lists to store predicted and actual tags
            predict = []
            count = 0

            cap = cv2.VideoCapture(dhgy)  # capturing the video from the given path
            frameRate = cap.get(7)  # frame rate
            # removing all other files from the temp folder
            print(dhgy)
            files3 = glob('D:\\Python\\Pycharm\\PycharmProjects\\IPFinal\\temp\\temp_3\\*')
            for f3 in files3:
                os.remove(f3)
            files4 = glob('D:\\Python\\Pycharm\\PycharmProjects\\IPFinal\\temp\\temp_4\\*')
            for f4 in files4:
                os.remove(f4)
                print("Temp Emptied")
            while cap.isOpened():
                frameId = cap.get(1)  # current frame number
                ret, frame = cap.read()
                if ret != True:
                    break
                if frameId % math.floor(frameRate) == 0:
                    # storing the frames in a new folder named train_1
                    filename = 'D:\\Python\\Pycharm\\PycharmProjects\\IPFinal\\temp\\temp_3\\' + "_frame%d.jpg" % count
                    count += 1
                    cv2.imwrite(filename, frame)
                    img = cv2.imread(filename)
                    gamma = 1.1
                    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
                    img = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB)
                    imageObject = Image.fromarray(img)
                    sharpened1 = imageObject.filter(ImageFilter.SHARPEN)
                    sharpened2 = sharpened1.filter(ImageFilter.SHARPEN)
                    im2 = sharpened2.filter(ImageFilter.MedianFilter(size=3))
                    filename1 = 'D:\\Python\\Pycharm\\PycharmProjects\\IPFinal\\temp\\temp_4\\' + "_frame%d.jpg" % count
                    count += 1
                    im2.save(filename1)
            cap.release()
            print("While loop done")
            # reading all the frames from temp folder
            images = glob("D:\\Python\\Pycharm\\PycharmProjects\\IPFinal\\temp\\temp_4\\*.jpg")

            prediction_images = []
            for i in range(len(images)):
                img = image.load_img(images[i], target_size=(224, 224, 3))
                img = image.img_to_array(img)
                img = img / 255
                prediction_images.append(img)
            print("predict1 done")

            # converting all the frames for a test video into numpy array
            prediction_images = np.array(prediction_images)

            # opening and store file in a variable
            json_file = open('D:\\Python\\Pycharm\\PycharmProjects\\IPFinal\\model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            print("model created")
            # use Keras model_from_json to make a loaded model
            model = model_from_json(loaded_model_json)

            # load weights into new model
            model.load_weights("D:\\Python\\Pycharm\\PycharmProjects\\IPFinal\\modelw.hdf5")
            print("Loaded Model from disk")

            # compile and evaluate loaded model
            model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
            model.summary()
            print("model compiled")

            prediction = model.predict_classes(prediction_images)
            # appending the mode of predictions in predict list to assign the tag to the video
            predict.append(y.columns.values[s.mode(prediction)[0][0]])
            # appending the actual tag of the video
            print(predict)
            valu = predict[0]
            files1 = glob('D:\\Python\\Pycharm\\PycharmProjects\\IPFinal\\temp\\temp_3\\*')
            for f1 in files1:
                os.remove(f1)
            files2 = glob('D:\\Python\\Pycharm\\PycharmProjects\\IPFinal\\temp\\temp_4\\*')
            for f2 in files2:
                os.remove(f2)
            os.remove(dhgy)
            print("File emptied")
            return render_template('Solution1.html', name=valu)


if __name__ == '__main__':
    app.run(debug='true')

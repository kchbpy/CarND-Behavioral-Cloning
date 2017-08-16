'''
the model of drive_sim
'''
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import sklearn.utils as sk
import datetime

def generator(lines,batch_size=128):
    '''
    获取部分数据
    '''
    num_samples = len(lines)
    while 1:
        sk.shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples=lines[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sampl in batch_samples:
                imagepath = batch_sampl[0]
                center_image = cv2.imread(imagepath)
                center_image = cv2.resize(center_image,(160,80))
                angle = float(batch_sampl[3])
                images.append(center_image)
                angles.append(angle)
                left_image = cv2.imread(batch_sampl[1])
                left_image = cv2.resize(left_image,(160,80))
                images.append(left_image)
                angles.append(angle+0.2)
                right_image = cv2.imread(batch_sampl[2])
                right_image = cv2.resize(right_image,(160,80))
                images.append(right_image)
                angles.append(angle-0.2)

            X_train = np.array(images)/255.0-0.5
            Y_train = np.array(angles)
            yield sk.shuffle(X_train,Y_train)

lines = []
print("start")
print(datetime.datetime.now())
trainfiles = ["example//driving_log.csv", "train1//driving_log.csv", "train2//driving_log.csv", "train3//driving_log.csv"]
# trainfiles=["train1//driving.csv"]
for filename in trainfiles:
    with open('datas//'+filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

num_train = len(train_samples)
num_valid = len(validation_samples)

train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)


model = Sequential()
# model.add(Lambda(lambda x:x/255.0-0.5,input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((35, 12), (0, 0)), input_shape=(80, 160, 3)))
# model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Convolution2D(24, (3, 3),strides=(2, 2),activation="relu"))
model.add(Convolution2D(36, (3, 3),strides=(2, 2),activation="relu"))
model.add(Convolution2D(48, (3, 3),strides=(2, 2),activation="relu"))
model.add(Convolution2D(64, (2, 2),strides=(1, 1),activation="relu"))
model.add(Convolution2D(64, (2, 2),strides=(1, 1),activation="relu"))
model.add(Convolution2D(30,kernel))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
model.fit_generator(train_generator, num_train, epochs = 4, validation_data = validation_generator, validation_steps = num_valid, verbose = 2)

model.save("model.h5")

print("end")
print(datetime.datetime.now())


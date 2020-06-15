import tensorflow as tf
from keras.preprocessing.image import img_to_array
import cv2
from keras import models
from keras import layers
import keras
import numpy
import matplotlib

inputVid=cv2.VideoCapture('Bee__2825.mp4')
frameCount=0
while inputVid.isOpened():
    ret, frame = inputVid.read()
    if ret == False:
        break
    frameCount+=1
    path="Image/"+str(frameCount)+".jpg"
    cv2.imwrite(path)


x=layers.Input(batch_shape=(784))
edl1=layers.Dense(units=300)(x)
ea1=layers.LeakyReLU(alpha=0.3)(edl1)
edl2=layers.Dense(units=2)(ea1)
eo=layers.LeakyReLU()(edl2)
encoder=models.Model(x,eo)

di=layers.Input(shape=(2))
ddl1=layers.Dense(units=300)(di)
dal1=layers.LeakyReLU()(ddl1)
ddl2=layers.Dense(units=784)(dal1)
do=layers.LeakyReLU()(ddl2)
decoder=models.Model(di,do)

aei=layers.Input(batch_shape=(784))
aeeo=encoder(aei)
aedo=decoder(aeeo)

ae=models.Model(aei,aedo)

ae.compile(loss='mse',optimizer=keras.optimizers.Adam(lr=0.0005))


encoded_images = encoder.predict(x_train)
decoded_images = decoder.predict(encoded_images)
decoded_images_orig = numpy.reshape(decoded_images, newshape=(decoded_images.shape[0], 28, 28))

num_images_to_show=5


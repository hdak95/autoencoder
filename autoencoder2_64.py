import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input,Dense,Conv2D,Conv2DTranspose,MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.models import Model
import matplotlib.pyplot as plt
import time

#cifar10 dataset load
(x_train,_),(x_test,_)=keras.datasets.cifar10.load_data()

#gaussian noise generate
noise1=np.random.normal(0,10,x_train.shape)
noise2=np.random.normal(0,10,x_test.shape)

#noise image normalization
x_train_noise=(x_train+noise1) / 255.0
x_test_noise=(x_test+noise2) / 255.0

#test normalization
x_test=x_test / 255.0

input_img=Input(shape=(32,32,3))

#Encoder
x=Conv2D(64,(3,3),activation='relu',padding='same')(input_img)  #32x32x64
x=MaxPooling2D((2,2),padding='same')(x) #16x16x64
x=Conv2D(64,(3,3),activation='relu',padding='same')(x) #16x16x64
x=MaxPooling2D((2,2),padding='same')(x) #8x8x64
x=Conv2D(64,(3,3),activation='relu',padding='same')(x)  #8x8x64
encoded=MaxPooling2D((2,2),padding='same',name='encoder')(x)  #latent vector 4x4x64

#Decoder
x=Conv2DTranspose(64,(3,3),strides=(2,2),activation='relu',padding='same')(encoded) #8x8x64
x=Conv2DTranspose(64,(3,3),strides=(2,2),activation='relu',padding='same')(x) #16x16x64
decoded=Conv2DTranspose(3,(3,3),strides=(2,2),activation='sigmoid',padding='same')(x) #output 32x32x3 

#autoencoder
autoencoder=Model(input_img,decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam',loss='mse')

#model fitting
autoencoder.fit(x_train_noise,x_train_noise,
          epochs=10,
          batch_size=64,
          shuffle=True,
          validation_data=(x_test_noise,x_test_noise))

#inference time
start = time.time()

#auto encoder based noise reduction
predicted=autoencoder.predict(x_test_noise)
print("time :", time.time() - start)

import math
#psnr function
def psnr(src,dst):
  mse=np.mean((src-dst)**2)
  if mse==0:
    return 100
  max=1
  return 10*math.log10(max/mse)

#reduction image psnr value
psnr_val=[]
for i in range(len(predicted)):
  psnr_val.append(psnr(predicted[i],x_test[i]))

print("psnr max: ",max(psnr_val))
print("psnr min: ",min(psnr_val))
print("psnr mean: ",sum(psnr_val)/len(psnr_val))

#display original / noise added / reduction image
plt.figure(figsize=(40,4))
for i in range(10):
  ax=plt.subplot(3,20,i+1)
  plt.imshow(x_test[i].reshape(32,32,3))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  ax=plt.subplot(3,20,20+i+1)
  plt.imshow(x_test_noise[i].reshape(32,32,3))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  ax=plt.subplot(3,20,40+i+1)
  plt.imshow(predicted[i].reshape(32,32,3))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

plt.show()

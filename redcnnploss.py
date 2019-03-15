'''
redcnn:
Hu Chen, Yi Zhang*, Member, IEEE, Mannudeep K. Kalra, Feng Lin, Yang Chen, Peixo Liao, Jiliu Zhou,
Senior Member, IEEE, and Ge Wang, Fellow, IEEE
Implemented by Wenxuan Kang
'''

from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import keras.backend as K
from keras.models import Model
from keras import layers
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.optimizers import Adam
from vgg19_ploss import vgg19ploss
from vgg19_ploss import perceptual_loss
import numpy as np




def red_cnn():
    input_img = Input(shape=(600, 600, 3), name='input_img')
    conv1 = Conv2D(32, (3, 3), strides=(1, 1), use_bias=False, name='conv1', padding='same', init='uniform')(input_img)
    en_relu1 = Activation('relu', name='en_relu1')(conv1)
    conv2 = Conv2D(32, (3, 3), strides=(1, 1), use_bias=False, name='conv2', padding='same', init='uniform')(en_relu1)
    en_relu2 = Activation('relu', name='en_relu2')(conv2)
    conv3 = Conv2D(32, (3, 3), strides=(1, 1), use_bias=False, name='conv3', padding='same', init='uniform')(en_relu2)
    en_relu3 = Activation('relu', name='en_relu3')(conv3)
    conv4 = Conv2D(32, (3, 3), strides=(1, 1), use_bias=False, name='conv4', padding='same', init='uniform')(en_relu3)
    en_relu4 = Activation('relu', name='en_relu4')(conv4)
    conv5 = Conv2D(32, (3, 3), strides=(1, 1), use_bias=False, name='conv5', padding='same', init='uniform')(en_relu4)
    en_relu5 = Activation('relu', name='en_relu5')(conv5)

    conv_Transpose1 = Conv2DTranspose(32, (3, 3), strides=(1, 1), use_bias=False, name='conv_Transpose1', padding='same',
                                      init='uniform')(en_relu5)
    conv_Transpose1 = layers.add([conv_Transpose1, en_relu4])
    de_relu1 = Activation('relu', name='de_relu1')(conv_Transpose1)

    conv_Transpose2 = Conv2DTranspose(32, (3, 3), strides=(1, 1), use_bias=False, name='conv_Transpose2', padding='same',
                                      init='uniform')(de_relu1)
    de_relu2 = Activation('relu', name='de_relu2')(conv_Transpose2)

    conv_Transpose3 = Conv2DTranspose(32, (3, 3), strides=(1, 1), use_bias=False, name='conv_Transpose3', padding='same',
                                      init='uniform')(de_relu2)
    conv_Transpose3 = layers.add([conv_Transpose3, en_relu2])
    de_relu3 = Activation('relu', name='de_relu3')(conv_Transpose3)

    conv_Transpose4 = Conv2DTranspose(32, (3, 3), strides=(1, 1), use_bias=False, name='conv_Transpose4', padding='same',
                                      init='uniform')(de_relu3)
    de_relu4 = Activation('relu', name='de_relu4')(conv_Transpose4)

    conv_Transpose5 = Conv2DTranspose(1, (3, 3), strides=(1, 1), use_bias=False, name='conv_Transpose5', padding='same',
                                      init='uniform')(de_relu4)
    conv_Transpose5 = layers.add([conv_Transpose5, input_img])
    de_relu5 = Activation('relu', name='de_relu5')(conv_Transpose5)

    model = Model(input_img, de_relu5, name='red-cnn')

    return model



model = red_cnn()
adam = Adam(lr=1e-4)
model.compile(optimizer=adam,
              loss=perceptual_loss)
data = np.load("C:/Users/DELL/PycharmProjects/bisenet/data_3_224.npy")
labels = np.load("C:/Users/DELL/PycharmProjects/bisenet/data_6_224.npy")
model.fit(data, labels, epochs=50, batch_size=4,validation_split=0)
model.save('redcnn4.h5')
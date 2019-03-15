from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
import warnings
import numpy as np
import keras
from keras.preprocessing import image
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
from keras.layers import AveragePooling2D
from keras.layers.merge import Concatenate
import numpy as np
from keras.layers.core import Flatten
from losses import perceptual_loss
from keras import losses

config = tf.ConfigProto(allow_soft_placement=True,\
        device_count = {'CPU' : 1, 'GPU' : 1})
session = tf.Session(config=config)
K.set_session(session)


def CTDNgan(shape):
    '''
    生成器（generator）
    red-cnn
    '''
    generator_input = Input(shape=(shape, shape, 3), name='input_img')
    conv1 = Conv2D(32, (3, 3), strides=(1, 1), use_bias=False, name='conv1', padding='same', init='uniform')(generator_input)
    en_relu1 = Activation('relu', name='en_relu1')(conv1)
    conv2 = Conv2D(32, (3, 3), strides=(1, 1), use_bias=False, name='conv2', padding='same', init='uniform')(en_relu1)
    en_relu2 = Activation('relu', name='en_relu2')(conv2)
    conv3 = Conv2D(32, (3, 3), strides=(1, 1), use_bias=False, name='conv3', padding='same', init='uniform')(en_relu2)
    en_relu3 = Activation('relu', name='en_relu3')(conv3)
    conv4 = Conv2D(32, (3, 3), strides=(1, 1), use_bias=False, name='conv4', padding='same', init='uniform')(en_relu3)
    en_relu4 = Activation('relu', name='en_relu4')(conv4)
    conv5 = Conv2D(32, (3, 3), strides=(1, 1), use_bias=False, name='conv5', padding='same', init='uniform')(en_relu4)
    en_relu5 = Activation('relu', name='en_relu5')(conv5)

    conv_Transpose1 = Conv2DTranspose(32, (3, 3), strides=(1, 1), use_bias=False, name='conv_Transpose1',
                                      padding='same',
                                      init='uniform')(en_relu5)
    conv_Transpose1 = layers.add([conv_Transpose1, en_relu4])
    de_relu1 = Activation('relu', name='de_relu1')(conv_Transpose1)

    conv_Transpose2 = Conv2DTranspose(32, (3, 3), strides=(1, 1), use_bias=False, name='conv_Transpose2',
                                      padding='same',
                                      init='uniform')(de_relu1)
    de_relu2 = Activation('relu', name='de_relu2')(conv_Transpose2)

    conv_Transpose3 = Conv2DTranspose(32, (3, 3), strides=(1, 1), use_bias=False, name='conv_Transpose3',
                                      padding='same',
                                      init='uniform')(de_relu2)
    conv_Transpose3 = layers.add([conv_Transpose3, en_relu2])
    de_relu3 = Activation('relu', name='de_relu3')(conv_Transpose3)

    conv_Transpose4 = Conv2DTranspose(32, (3, 3), strides=(1, 1), use_bias=False, name='conv_Transpose4',
                                      padding='same',
                                      init='uniform')(de_relu3)
    de_relu4 = Activation('relu', name='de_relu4')(conv_Transpose4)

    conv_Transpose5 = Conv2DTranspose(3, (3, 3), strides=(1, 1), use_bias=False, name='conv_Transpose5', padding='same',
                                      init='uniform')(de_relu4)
    conv_Transpose5 = layers.add([conv_Transpose5, generator_input])
    de_relu5 = Activation('relu', name='de_relu5')(conv_Transpose5)


    generator = Model(generator_input, de_relu5)
    generator.summary()
    '''
    discriminator(鉴别器)
    多尺度特征融合
    '''
    discriminator_input = Input(shape=(shape, shape, 3), name='input_img')
    x1 = Conv2D(32, (3, 3), strides=(1, 1), use_bias=False, name='block1_conv33', padding='same', init='uniform')(
        discriminator_input)
    x1 = BatchNormalization(name='block1_conv33_bn')(x1)
    x1 = Activation('relu', name='block1_conv33_relu')(x1)

    x2 = Conv2D(32, (5, 1), strides=(1, 1), use_bias=False, name='block1_conv51', padding='same', init='uniform')(
        discriminator_input)
    x2 = Conv2D(32, (1, 5), strides=(1, 1), use_bias=False, name='block1_conv15', padding='same', init='uniform')(
        x2)
    x2 = BatchNormalization(name='block1_conv55_bn')(x2)
    x2 = Activation('relu', name='block1_conv55_relu')(x2)

    x3 = Conv2D(32, (7, 1), strides=(1, 1), use_bias=False, name='block1_conv71', padding='same', init='uniform')(
        discriminator_input)
    x3 = Conv2D(32, (1, 7), strides=(1, 1), use_bias=False, name='block1_conv17', padding='same', init='uniform')(
        x3)
    x3 = BatchNormalization(name='block1_conv77_bn')(x3)
    x3 = Activation('relu', name='block1_conv77_relu')(x3)

    x4 = Concatenate(axis=3)([x1, x2, x3])
    x4 = Conv2D(32, (1, 1), strides=(1, 1), use_bias=False, name='block2_conv11', padding='same', init='uniform')(
        x4)
    x4 = BatchNormalization(name='block2_conv11_bn')(x4)
    x4 = Activation('relu', name='block2_conv11_relu')(x4)

    x4 = AveragePooling2D(name='avg_pool_x4', data_format='channels_last',
                          pool_size=(int(x4.shape[1]), int(x4.shape[2])))(x4)

    x4 = Flatten()(x4)
    x4 = Dense(1, activation='sigmoid', name='doutput')(x4)
    discriminator = Model(discriminator_input, x4)
    discriminator.summary()
    discriminator_optimizer = keras.optimizers.RMSprop(lr=8e-4, clipvalue=1.0, decay=1e-8)
    discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
    '''
    The adversarial network
    
    '''
    # 将鉴别器（discrimitor）权重设置为不可训练（仅适用于`gan`模型）
    discriminator.trainable = False

    gan_input = keras.Input(shape=(shape, shape, 3))
    goutput = keras.layers.core.Reshape(target_shape=(shape, shape, 3), name='goutput')(generator(gan_input))
    gan_output = keras.layers.core.Reshape(target_shape=[1], name='doutput')(discriminator(goutput))
    gan = keras.models.Model(gan_input, outputs=[goutput, gan_output])

    gan_optimizer = keras.optimizers.RMSprop(lr=4e-4, clipvalue=1.0, decay=1e-8)
    gan.compile(optimizer=gan_optimizer, loss={'goutput': perceptual_loss, 'doutput': 'binary_crossentropy'},loss_weights={'goutput': 1., 'doutput': 1})

    return generator, discriminator, gan

import os
from keras.preprocessing import image

generator, discriminator, gan = CTDNgan(640)

data = np.load("C:/Users/DELL/PycharmProjects/bisenet/data_3.npy")
labels = np.load("C:/Users/DELL/PycharmProjects/bisenet/data_6.npy")
test_data = data[-10:-1]
test_labels = labels[-10:-1]
iterations = 200
batch_size = 1


start = 0
if __name__ == '__main__':
    for step in range(iterations):
        stop = start + batch_size
        high_imgs = labels[start: stop]
        low_imgs = data[start: stop]

        generated_images = generator.predict(low_imgs)

        # 将假图像与真实图像进行比较

        combined_images = np.concatenate([generated_images, high_imgs])

        # 组装区别真假图像的标签
        label = np.concatenate([np.ones((batch_size, 1)),
                                 np.zeros((batch_size, 1))])
        # 重要的技巧，在标签上添加随机噪声
        label += 0.05 * np.random.random(label.shape)

        # 训练鉴别器（discrimitor）
        d_loss = discriminator.train_on_batch(combined_images, label)
        td_loss = losses.binary_crossentropy(discriminator.predict(test_data), discriminator.predict(test_labels))
        # 汇集标有“所有高质量图像”的标签
        misleading_targets = np.zeros((batch_size, 1))

        # 训练生成器（generator）（通过gan模型，鉴别器（discrimitor）权值被冻结）
        a_loss = gan.train_on_batch(low_imgs, [high_imgs, misleading_targets])

        #test_gen = gan(inputs=gan.input, outputs=gan.get_layer('goutput').output)
        #ta_losses = perceptual_loss(tf.convert_to_tensor(test_gen.predict(test_data), np.float32),
        #                                       tf.convert_to_tensor(test_gen.predict(test_labels), np.float32))
        #with tf.Session() as sess:
        #    ta_loss = K.mean(ta_losses).eval()

        start += batch_size
        if start > len(data) - batch_size - 10:
            start = 0
        if step % 10 == 0:
            # 保存网络权值
            gan.save_weights('gan_ploss1.h5')

            # 输出metrics
            print('discriminator train loss at step %s: %s' % (step, d_loss))
            print('adversarial train loss at step %s: %s' % (step, a_loss))
            print('discriminator test loss at step %s: %s' % (step, td_loss))
            #print('adversarial loss at step %s: %s' % (step, a_loss))
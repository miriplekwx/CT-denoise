from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
from PIL import Image
from keras import backend as K
import os
import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
#gpu_options = tf.GPUOptions(allow_growth=True)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


#【0】VGG19模型，加载预训练权重
base_model = VGG19(weights='imagenet')

#【1】创建一个新model, 使得它的输出(outputs)是 VGG19 中任意层的输出(output)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)


def vgg19pdet(img1, img2):
    img1 = image.load_img(img1, target_size=(224, 224))  # 加载图片并resize成224x224
    img2 = image.load_img(img2, target_size=(224, 224))  # 加载图片并resize成224x224

    img1 = img1.resize((224, 224), Image.ANTIALIAS)
    img1 = image.img_to_array(img1)
    img1 = np.expand_dims(img1, axis=0)
    img1 = preprocess_input(img1)
    img2 = img2.resize((224, 224), Image.ANTIALIAS)
    img2 = image.img_to_array(img2)
    img2 = np.expand_dims(img2, axis=0)
    img2 = preprocess_input(img2)
    print(img2.shape)
    features1 = model.predict(img1)#(1, 14, 14, 512)
    features2 = model.predict(img2)#(1, 14, 14, 512)

    pmse = K.mean(K.mean(K.mean(K.square(features1 - features2), axis=-1), axis=-1), axis=-1)
    return pmse.eval(session=tf.Session())

def vgg19pdet_img(img1, img2):

    #img1 = img1.resize((224, 224), Image.ANTIALIAS)
    #img1 = image.img_to_array(img1)
    img1 = np.expand_dims(img1, axis=0)
    img1 = preprocess_input(img1)
    #img2 = img2.resize((224, 224), Image.ANTIALIAS)
    #img2 = image.img_to_array(img2)
    img2 = np.expand_dims(img2, axis=0)
    img2 = preprocess_input(img2)
    print(img2.shape)
    features1 = model.predict(img1)#(1, 14, 14, 512)
    features2 = model.predict(img2)#(1, 14, 14, 512)

    pmse = K.mean(K.mean(K.mean(K.square(features1 - features2), axis=-1), axis=-1), axis=-1)
    return pmse.eval(session=tf.Session())

def vgg19ploss(img1, img2):
    '''
    img1 = Image.fromarray(img1[0].astype('uint8')).convert('RGB')
    img2 = Image.fromarray(img2[0].astype('uint8')).convert('RGB')
    img1 = img1.resize((224, 224), Image.ANTIALIAS)
    img1 = image.img_to_array(img1)
    img1 = np.expand_dims(img1, axis=0)
    img1 = preprocess_input(img1)
    img2 = img2.resize((224, 224), Image.ANTIALIAS)
    img2 = image.img_to_array(img2)
    img2 = np.expand_dims(img2, axis=0)
    img2 = preprocess_input(img2)
    '''
    #f K.int_shape(img1)[0] == None or K.int_shape(img2)[0] == None:
    return K.mean(K.mean(K.mean(K.square(img1 - img2), axis=-1), axis=-1), axis=-1)
    #features1 = model.predict(img1)#(batch, 14, 14, 512)
    #features2 = model.predict(img2)#(batch, 14, 14, 512)


    #pmse = K.mean(K.square(features1 - features2), axis=-1)
    #return pmse


def perceptual_loss(img_true, img_generated):
    image_shape = (224, 224, 3)
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_block3 = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_block3.trainable = False
    loss_block2 = Model(inputs=vgg.input, outputs=vgg.get_layer('block2_conv2').output)
    loss_block2.trainable = False
    loss_block1 = Model(input=vgg.input, outputs=vgg.get_layer('block1_conv2').output)
    loss_block1.trainable = False
    return K.mean(K.square(loss_block1(img_true) - loss_block1(img_generated))) + 2 * K.mean(
        K.square(loss_block2(img_true) - loss_block2(img_generated))) + 5 * K.mean(
        K.square(loss_block3(img_true) - loss_block3(img_generated)))
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
from PIL import Image
from keras import backend as K
import tensorflow as tf



def perceptual_loss(img_true, img_generated):
    image_shape = (600, 600, 3)
    xception = Xception(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_block3 = Model(inputs=xception.input, outputs=xception.get_layer('block5_sepconv1_act').output)
    loss_block3.trainable = False
    #loss_block2 = Model(inputs=xception.input, outputs=xception.get_layer('block2_conv2').output)
    #loss_block2.trainable = False
    #loss_block1 = Model(input=xception.input, outputs=xception.get_layer('block1_conv2').output)
    #loss_block1.trainable = False
    #return K.mean(K.square(loss_block1(img_true) - loss_block1(img_generated))) + 2 * K.mean(
    #    K.square(loss_block2(img_true) - loss_block2(img_generated))) + 5 * K.mean(
    #    K.square(loss_block3(img_true) - loss_block3(img_generated)))
    return K.mean(K.square(loss_block3(img_true) - loss_block3(img_generated)))



def vgg19pdet(img1, img2):
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
    img1 = image.load_img(img1, target_size=(224, 224))
    img2 = image.load_img(img2, target_size=(224, 224))
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

if __name__ == '__main__':
    image_shape = (600, 600, 3)
    xception = Xception(include_top=False, weights='imagenet', input_shape=image_shape)
    from keras.utils.vis_utils import plot_model
    plot_model(xception, to_file='model1.png', show_shapes=True)




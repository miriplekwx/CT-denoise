import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import load_model
from keras import Model
from PIL import Image
from keras.preprocessing import image
import cv2
from vgg19_ploss import perceptual_loss
import qutofimg
from vgg19_ploss import vgg19pdet_img
import ganfirst

q, w, model = ganfirst.CTDNgan(224)
model.load_weights('gan_ploss.h5')


model_pre = Model(inputs=model.input, outputs=model.get_layer('goutput').output)

#from keras.utils.vis_utils import plot_model
#plot_model(model_pre, to_file='model3.png', show_shapes=True)





img_path = 'C://Users/DELL/Desktop//ct_data_jpg//3/320_3.jpg'
img1 = cv2.imread("C://Users/DELL/Desktop//ct_data_jpg//3/320_3.jpg", cv2.IMREAD_COLOR)
img1 = cv2.resize(img1, (224, 224), interpolation=cv2.INTER_CUBIC)
img6 = cv2.imread("C:/Users/DELL/Desktop/ct_data_jpg/6/320_6.jpg", cv2.IMREAD_COLOR)
img6 = cv2.resize(img6, (224, 224), interpolation=cv2.INTER_CUBIC)

#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
x1 = np.expand_dims(img1, axis=0)
x6 = np.expand_dims(img6, axis=0)
#x = preprocess_input(x)
#print('Input image shape:', x1.shape)


pre = model.predict(x1)
pre = np.array(pre[0])
print(pre.shape)


#cv2.imshow("image0", x1[0])
#cv2.imshow("image1", pre[0]/255)
#cv2.imshow("image6", x6[0])

c = (pre[0]-x1[0])/1.5
cv2.imshow("image2", c)
cv2.waitKey(0)


#print(vgg19pdet_img(pre[0],img6), vgg19pdet_img(img1,img6))
#print(qutofimg.vifp_mscale(pre,img6),qutofimg.vifp_mscale(img1,img6))
#print(qutofimg.inf_entr_det(pre[0]),qutofimg.inf_entr_det(img1),qutofimg.inf_entr_det(img6))
#print(qutofimg.psnr(pre[0],img6),qutofimg.psnr(img1,img6))
#print(qutofimg.ave_diff_det(pre[0]),qutofimg.ave_diff_det(img1),qutofimg.ave_diff_det(img6))

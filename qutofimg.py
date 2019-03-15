import cv2
import numpy as np
import math
import scipy.signal
import scipy.ndimage
from scipy.ndimage import gaussian_filter
from vgg19_ploss import vgg19pdet
from numpy.lib.stride_tricks import as_strided as ast
from keras.preprocessing import image

##平均梯度
def ave_diff_det(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1)
    sobely = cv2.convertScaleAbs(sobely)
    result = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    result_sum = np.sum(np.reshape(result,(result.size,)))/result.size
    return result_sum

##信息熵
def inf_entr_det(img):
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[int(val)] = float(tmp[int(val)] + 1)
            k =  float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res

#视觉信息保真度
def vifp_mscale(ref, dist):
    sigma_nsq = 2
    eps = 1e-10

    num = 0.0
    den = 0.0
    for scale in range(1, 5):

        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0

        if (scale > 1):
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den

    return vifp

#这个大家都知道
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

#结构相似性
def ssim(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):

    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return np.mean(ssim_map)

#这个也都知道
def mse_det(img1, img2):
    img = img1-img2
    return np.linalg.norm(img)


if __name__ == '__main__':

    img2 = cv2.imread("C:/Users/DELL/Desktop/ct_data_jpg/2/1_2.jpg", cv2.IMREAD_UNCHANGED)
    img25 = cv2.imread("C:/Users/DELL/Desktop/ct_data_jpg/2.5/1_2.5.jpg", cv2.IMREAD_UNCHANGED)
    img3 = cv2.imread("C:/Users/DELL/Desktop/ct_data_jpg/3/1_3.jpg", cv2.IMREAD_UNCHANGED)
    img35 = cv2.imread("C:/Users/DELL/Desktop/ct_data_jpg/3.5/1_3.5.jpg", cv2.IMREAD_UNCHANGED)
    img4 = cv2.imread("C:/Users/DELL/Desktop/ct_data_jpg/4/1_4.jpg", cv2.IMREAD_UNCHANGED)
    img45 = cv2.imread("C:/Users/DELL/Desktop/ct_data_jpg/4.5/1_4.5.jpg", cv2.IMREAD_UNCHANGED)
    img5 = cv2.imread("C:/Users/DELL/Desktop/ct_data_jpg/5/1_5.jpg", cv2.IMREAD_UNCHANGED)
    img55 = cv2.imread("C:/Users/DELL/Desktop/ct_data_jpg/5.5/1_5.5.jpg", cv2.IMREAD_UNCHANGED)
    img6 = cv2.imread("C:/Users/DELL/Desktop/ct_data_jpg/6/1_6.jpg", cv2.IMREAD_UNCHANGED)

    print(ave_diff_det(img25),ave_diff_det(img3),ave_diff_det(img35),ave_diff_det(img4),ave_diff_det(img45),ave_diff_det(img5),ave_diff_det(img55),ave_diff_det(img6))
    #
    print(inf_entr_det(img25),inf_entr_det(img3),inf_entr_det(img35),inf_entr_det(img4),inf_entr_det(img45),inf_entr_det(img5),inf_entr_det(img55),inf_entr_det(img6),)
    #
    print(vifp_mscale(img6, img25),vifp_mscale(img6, img3),vifp_mscale(img6, img35),vifp_mscale(img6, img4),vifp_mscale(img6, img45),vifp_mscale(img6, img5),vifp_mscale(img6, img55),vifp_mscale(img6, img6),)
    #
    print(psnr(img6, img25),psnr(img6, img3),psnr(img6, img35),psnr(img6, img4),psnr(img6, img45),psnr(img6, img5),psnr(img6, img55),psnr(img6, img6))
    print(ssim(img6, img25),ssim(img6, img3),ssim(img6, img35),ssim(img6, img4),ssim(img6, img45),ssim(img6, img5),ssim(img6, img55),ssim(img6, img6))
    print(mse_det(img6, img25),mse_det(img6, img3),mse_det(img6, img35),mse_det(img6, img4),mse_det(img6, img45),mse_det(img6, img5),mse_det(img6, img55),mse_det(img6, img6))

    im25 = 'C:/Users/DELL/Desktop/ct_data_jpg/2.5/1_2.5.jpg'
    im3 = 'C:/Users/DELL/Desktop/ct_data_jpg/3/1_3.jpg'
    im35 = 'C:/Users/DELL/Desktop/ct_data_jpg/3.5/1_3.5.jpg'
    im4 = 'C:/Users/DELL/Desktop/ct_data_jpg/4/1_4.jpg'
    im45 = 'C:/Users/DELL/Desktop/ct_data_jpg/4.5/1_4.5.jpg'
    im5 = 'C:/Users/DELL/Desktop/ct_data_jpg/5/1_5.jpg'
    im55 = 'C:/Users/DELL/Desktop/ct_data_jpg/5.5/1_5.5.jpg'
    im6 = 'C:/Users/DELL/Desktop/ct_data_jpg/6/1_6.jpg'

    print(vgg19pdet(im25,im6),vgg19pdet(im3,im6),vgg19pdet(im35,im6),vgg19pdet(im4,im6),vgg19pdet(im45,im6),vgg19pdet(im5,im6),vgg19pdet(im55,im6),vgg19pdet(im6,im6))


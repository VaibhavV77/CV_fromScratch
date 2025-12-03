import cv2
import numpy as np
img= cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
def del2_gauss(size,sigma):
    filt=np.ones((size,size),dtype=float)
    x=size//2
    y=size//2
    for i in range(size):
        for j in range(size):
            filt[i, j] = ((((i-x)**2+(j-y)**2-2*sigma**2)/(sigma**4))*np.exp(-((i-x)**2+(j-y)**2)/(2*sigma**2))).astype(np.float32)
    return filt
def convolve(filt,img):
    h,w=img.shape
    a,b=filt.shape
    pad=a//2
    img= np.pad(img, pad, mode='constant',constant_values=0)
    imgF = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
       for j in range(w):
           conv=(img[i:i+a,j:j+b]*filt).astype(np.float32)
           imgF[i,j]=np.sum(conv).astype(np.float32)
    return imgF
lap_kern=del2_gauss(5,1)
edges=convolve(lap_kern,img)
cv2.imshow('edge detection by laplacian',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
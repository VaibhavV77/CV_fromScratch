import cv2
import numpy as np
img= cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
def gauss_Kern(size,sigma):
    x=size//2
    y=size//2
    h,w=img.shape
    kern=np.ones((size,size),dtype=float)
    for i in range(size):
        for j in range(size):
            kern[i][j]=np.exp(-1*((i-x)**2+(j-y)**2)/(2.*sigma**2))
    return kern/(2.*np.pi*sigma**2)



def convolve(filt,img):
    h,w=img.shape
    a,b=filt.shape
    pad=a//2
    img= np.pad(img, pad, mode='constant',constant_values=0)
    imgF = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
       for j in range(w):
           conv=img[i:i+a,j:j+b]*filt
           imgF[i,j]=np.sum(conv)
    return np.clip(imgF, 0, 255).astype(np.uint8)

def sobel_Kern(img):
    kern_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float32)
    kern_y=np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=np.float32)
    Grad_x=convolve(kern_x,img).astype(np.float32)
    Grad_y=convolve(kern_y,img).astype(np.float32)
    edges = np.sqrt(Grad_x**2+Grad_y**2)
    edges = (edges/edges.max() * 255).astype(np.uint8)
    _, binary_edges = cv2.threshold(edges,20,255,cv2.THRESH_BINARY)

    cv2.imshow('sobel detection', binary_edges)

GSimg=convolve(gauss_Kern(5,1),img)
cv2.imshow('smoothed image',GSimg)
sobel_Kern(GSimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np
img1=cv2.imread('Screenshot 2025-04-14 114151.png')
img2=cv2.imread('Screenshot 2025-04-14 114222.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
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
def draw_blobs(image, blobs):
    for y,x in blobs:
        cv2.circle(image,(int(x),int(y)),4,(0,0,255),2)
    return image
def blob_detect(GSimg,sigmas):
    h,w=GSimg.shape
    blobs=np.zeros((len(sigmas),h,w),dtype=np.float32)
    for i,sigma in enumerate(sigmas):
        size=int(6*sigma)|1
        filt=del2_gauss(size,sigma)
        blobs[i]=np.abs(convolve(filt,GSimg))*sigma**2
    return blobs
        
def minima_3d(scl,sigmas,thresh):
    sig,h,w=scl.shape
    minima=[]
    for s in range(1,sig-1):
      for i in range(1,h-1):
          for j in range(1,w-1):
            center=scl[s,i,j]
            neighborhood=scl[s-1:s+2,i-1:i+2,j-1:j+2]
            if center<thresh:
                continue
            if center==np.min(neighborhood) and np.count_nonzero(neighborhood==center)==1:
                    minima.append((i,j))
    return minima
def remove_excBlob(blobs, min_dist=10):
    filtered = []
    for b in blobs:
        keep = True
        for fb in filtered:
            dist = np.sqrt((b[0]-fb[0])**2 + (b[1]-fb[1])**2)
            if dist < min_dist:
                keep = False
                break
        if keep:
            filtered.append(b)
    return filtered
sigmas = [1.2, 1.6, 2.0, 2.5, 3.0]
thresh=0.05
blobs1=blob_detect(gray1, sigmas)
minimas1 = minima_3d(blobs1,sigmas,thresh)
minimas1=remove_excBlob(minimas1, min_dist=10)
blobs1= draw_blobs(img1.copy(), minimas1)
cv2.imshow("Blobs1",blobs1)
blobs2=blob_detect(gray2,sigmas)
minimas2 = minima_3d(blobs2,sigmas,thresh)
minimas2=remove_excBlob(minimas2,min_dist=10)
blobs2= draw_blobs(img2.copy(),minimas2)
cv2.imshow("Blobs2",blobs2)
cv2.waitKey(0)
cv2.destroyAllWindows()

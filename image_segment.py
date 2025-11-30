import numpy as np
import cv2

image_gray = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)

_, img=cv2.threshold(image_gray,220,255,cv2.THRESH_BINARY)
img=255-img
cv2.imshow('img',img)
h,w=img.shape
def med_filt(image, f_size=5):
   if f_size% 2==0:
        raise ValueError("Give a odd k")
   h,w = image.shape
   pad=f_size//2
   p_img= np.pad(image, pad, mode='constant',constant_values=0)
   imgF = np.zeros_like(image)
   for i in range(h):
       for j in range(w):
           filt=p_img[i:i+f_size,j:j+f_size]
           imgF[i,j]=np.median(filt)
   return imgF     
imgF=med_filt(img,f_size=5)
cv2.imshow('filtered image',imgF)
imgF=imgF//255
def img_segment(image):
    h,w=image.shape
    img_label=np.zeros_like(image,dtype=int)
    equ_dict={}
    c=1
    for i in range(h):
       for j in range(w):
           top=img_label[i][j-1] if j>0 else 0
           left=img_label[i-1][j] if i>0 else 0
           if image[i][j]==1:
               if top==0 and left>0:
                   img_label[i][j]=left
               elif left==0 and top>0:
                   img_label[i][j]=top
               elif left==0 and top==0:
                   img_label[i][j]=c
                   equ_dict[c]=c
                   c=c+1
               else:
                   if(left==top):
                       img_label[i][j]=left
                   else:
                       img_label[i, j] = min(left, top)
                       equ_dict[max(left,top)]=min(left,top)
    def lab(label):
        while equ_dict[label]!=label:
            label = equ_dict[label]
        return label
    
    for i in range(h):
        for j in range(w):
            if img_label[i,j] > 0:
                img_label[i,j]=lab(img_label[i, j])
    unq_labels=np.unique(img_label[img_label > 0])
    label_map={old:new for new,old in enumerate(unq_labels,start=1)}
    
    for i in range(h):
        for j in range(w):
            if img_label[i, j] in label_map:
                img_label[i,j]=label_map[img_label[i,j]]

    return img_label

            

labels=img_segment(imgF)
print(labels)
cv2.imshow('Labeled Image',(labels/labels.max()*255).astype(np.uint8)) 
cv2.waitKey(0)
cv2.destroyAllWindows()
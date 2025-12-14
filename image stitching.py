import cv2
import numpy as np
import matplotlib.pyplot as plt

img1=cv2.imread('Screenshot 2025-04-14 114151.png')
img2=cv2.imread('Screenshot 2025-04-14 114222.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
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
def del2_gauss(size,sigma):
    filt=np.ones((size,size),dtype=float)
    x=size//2
    y=size//2
    for i in range(size):
        for j in range(size):
            filt[i, j] = ((((i-x)**2+(j-y)**2-2*sigma**2)/(sigma**4))*np.exp(-((i-x)**2+(j-y)**2)/(2*sigma**2))).astype(np.float32)
    return filt
def DoGset(img,s,sigma,k):
    scale_space=[]
    for i in range(k):
        filt=del2_gauss(5,sigma*(s**i))
        Simg=convolve(filt,img)
        scale_space.append(Simg)
    DoG=[]
    for i in range(k-1):
        DoG.append(scale_space[i+1]-scale_space[i])
    return DoG
def extremum(scl,thresh):
    minima = []
    for s in range(1,len(scl)-1):
        prev = scl[s-1]
        curr =scl[s]
        Next=scl[s+1]
        h,w=curr.shape
        for i in range(1,h-1):
            for j in range(1, w - 1):
                patch = np.array([prev[i-1:i+2,j-1:j+2],curr[i-1:i+2,j-1:j+2],Next[i-1:i+2,j-1:j+2]])
                val=curr[i,j]
                if abs(val)<=thresh:
                    continue
                if val==patch.max() or val==patch.min():
                    minima.append((j,i))
    return minima
DoG1 = DoGset(gray1,s=1.26,sigma=1.6,k=5)
SIFT1 = extremum(DoG1,thresh=20)
DoG2 = DoGset(gray2, s=1.26,sigma=1.6, k=5)
SIFT2 = extremum(DoG2,thresh=20)
vis1 = cv2.cvtColor(gray1,cv2.COLOR_GRAY2BGR)
vis2 = cv2.cvtColor(gray2,cv2.COLOR_GRAY2BGR)
for x,y in SIFT1:
    cv2.circle(vis1, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

for x, y in SIFT2:
    cv2.circle(vis2, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
cv2.imshow("SIFT1",vis1)
cv2.imshow("SIFT2",vis2)
def compute_dominant_orientation(mag, ori):
    hist = np.zeros(36)
    for i in range(16):
        for j in range(16):
            angle = ori[i, j]
            magnitude = mag[i, j]
            bin = int(angle//10)%36
            hist[bin] += magnitude
    dominant_angle = np.argmax(hist)*10
    return dominant_angle
def extract_patch(img, x, y, size=16):
    half = size // 2
    return img[y-half:y+half, x-half:x+half]
def mag_ORI(patch):
    Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float32)
    Gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=np.float32)
    grad_x = convolve(Gx,patch)
    grad_y = convolve(Gy,patch)
    mag = np.sqrt(grad_x**2+grad_y**2)
    ori = (np.arctan2(grad_y,grad_x)*180/np.pi)%360
    return mag, ori
def histComp(x,y,img):
    Descriptor=[]
    for i in range(x-8,x+8,4):
        for j in range(y-8,y+8,4):
            patch=img[j:j+4,i:i+4]
            mag,ori=mag_ORI(patch)
            hist = np.zeros(8)
            for m in range(4):
                for n in range(4):
                    angle=ori[m,n]
                    Mg=mag[m,n]
                    bin_idx=int(angle//45)%8
                    hist[bin_idx]+=Mg
            Descriptor.extend(hist)
    Descriptor=np.array(Descriptor)
    Descriptor = Descriptor / (np.linalg.norm(Descriptor)+1e-7)
    Descriptor = np.clip(Descriptor,0,0.2)
    Descriptor = Descriptor / (np.linalg.norm(Descriptor)+1e-7)
    return Descriptor
def match_descriptors(desc1,desc2,kp1,kp2):
    matches = []
    for i, d1 in enumerate(desc1):
        distances=[np.linalg.norm(d1-d2) for d2 in desc2]
        if len(distances) < 2:
            continue
        sorted_idx = np.argsort(distances)
        if distances[sorted_idx[0]] < 0.75 * distances[sorted_idx[1]]:
            x1, y1 = kp1[i]
            x2, y2 = kp2[sorted_idx[0]]
            matches.append((x1,y1,x2,y2))
    return matches
def inRange(x,y,img_shape,patch_radius=8):
    h, w = img_shape
    return (x - patch_radius >= 1 and y - patch_radius >= 1 and x + patch_radius < w - 1 and y + patch_radius < h - 1)

def desc(SIFT1,SIFT2,img1,img2):
    desc1=[]
    desc2=[]
    S1=[]
    S2=[]
    for x,y in SIFT1:
        if inRange(x, y,img1.shape,patch_radius=8):
           desc1.append(histComp(x,y,img1))
           S1.append((x,y))
    for x,y in SIFT2:
        if inRange(x, y,img2.shape,patch_radius=8):
           desc2.append(histComp(x,y,img2))
           S2.append((x,y))
    return desc1,desc2,S1,S2
desc1,desc2,S1,S2=desc(SIFT1,SIFT2,gray1,gray2)  

def draw_matches(img1, img2, matches):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    vis[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for (x1, y1, x2, y2) in matches:
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2 + w1), int(y2))
        cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
    return vis
matches = match_descriptors(desc1, desc2,S1,S2)
matched_img = draw_matches(gray1,gray2,matches)
cv2.imshow('matches',matched_img)
def Homography(matches):
    A=[]
    for i,(x1,y1,x2,y2) in enumerate(matches):
        A.append([x2,y2,1,0,0,0,-x1*x2,-x1*y2,-x1])
        A.append([0,0,0,x2,y2,1,-x2*y1,-y1*y2,-y1])
    A=np.array(A)
    Q=np.matmul(np.transpose(A),A)
    eigenvalues,eigenvectors=np.linalg.eig(Q)
    H = eigenvectors[:, np.argmin(eigenvalues)] 
    print(eigenvectors)
    H = H.reshape((3, 3))
    return H
H=Homography(matches)
print(H)
if H is not None:
    warped_img2 = cv2.warpPerspective(img2, H, (gray1.shape[1] + gray2.shape[1], max(gray1.shape[0], gray2.shape[0])))
    warped_img2[0:gray1.shape[0], 0:gray1.shape[1]] = img1
    cv2.imshow("Warped Image", warped_img2)
else:
    print("Homography computation failed.")
cv2.waitKey(0)
cv2.destroyAllWindows()

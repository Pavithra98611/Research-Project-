from scipy import ndimage
import imageio 
import numpy as np
import matplotlib.pyplot as plt
from skimage import color

img = imageio.imread("/content/drive/MyDrive/Colab Notebooks/Tripathaka.jpg")
img = img.astype('int32')
img = color.rgb2gray(img)
plt.imshow(img, cmap = plt.get_cmap('gray'))
plt.show()

#applying gaussian filter
img_gaussian_filter = ndimage.gaussian_filter(img, sigma = 1.4)
plt.imshow(img_gaussian_filter, cmap = plt.get_cmap('gray'))
plt.show()

#applying sobel filter
def SobelFilter(img, direction):
  if(direction == 'x'):
      Gx = np.array([[-1,0,+1], [-2,0,+2],  [-1,0,+1]])
      SImage = ndimage.convolve(img, Gx)
  if(direction == 'y'):
      Gy = np.array([[-1,-2,-1], [0,0,0], [+1,+2,+1]])
      SImage = ndimage.convolve(img, Gy)
  return SImage

def Normalize(img):
  Nimg = img/np.max(img)
  return Nimg

gx = SobelFilter(img_gaussian_filter, 'x')
gx = Normalize(gx)
gy = SobelFilter(img_gaussian_filter, 'y')
gy = Normalize(gy) 
plt.imshow(gx, cmap = plt.get_cmap('gray'))
plt.show()
plt.imshow(gy, cmap = plt.get_cmap('gray'))
plt.show()

Mag = np.hypot(gx,gy)
plt.imshow(Mag, cmap = plt.get_cmap('gray'))
plt.show()

Gmat = np.degrees(np.arctan2(gy,gx))
Gmat

def NonMaxSup(Gmag, Gmat):
   img = np.zeros(Gmag.shape)
   
   for i in range(1, int(Gmag.shape[0]) - 1):
        for j in range(1, int(Gmag.shape[1]) - 1):
            if((Gmat[i,j] >= -22.5 and Gmat[i,j] <= 22.5) or (Gmat[i,j] <= -157.5 and Gmat[i,j] >= 157.5)):
                if ((Gmag[i,j] > Gmag[i,j+1]) and (Gmag[i,j] > Gmag[i,j-1])):
                  img[i,j] = Gmag[i,j]
                else:
                    img[i,j] = 0

            if((Gmat[i,j] >= 22.5 and Gmat[i,j] <= 67.5) or (Gmat[i,j] <= -112.5 and Gmat[i,j] >= -157.5)):
                if ((Gmag[i,j] > Gmag[i+1,j+1]) and (Gmag[i,j] > Gmag[i-1,j-1])):
                  img[i,j] = Gmag[i,j]
                else:
                    img[i,j] = 0

            if((Gmat[i,j] >= 67.5 and Gmat[i,j] <= 112.5) or (Gmat[i,j] <= -67.5 and Gmat[i,j] >= -112.5)):
                if ((Gmag[i,j] > Gmag[i+1,j]) and (Gmag[i,j] > Gmag[i-1,j])):
                  img[i,j] = Gmag[i,j]
                else:
                    img[i,j] = 0
            if((Gmat[i,j] >= 112.5 and Gmat[i,j] <= 157.5) or (Gmat[i,j] <= -22.5 and Gmat[i,j] >= -67.5)):
                if ((Gmag[i,j] > Gmag[i+1,j-1]) and (Gmag[i,j] > Gmag[i-1,j+1])):
                  img[i,j] = Gmag[i,j]
                else:
                    img[i,j] = 0
      
   return img

img_NMS  = NonMaxSup(Mag,Gmat)
img_NMS = Normalize(img_NMS)
plt.imshow(img_NMS, cmap = plt.get_cmap('gray'))
plt.show()

def DoThreshHyst(img):
    highThresholdRatio =0.2
    lowThresholdRatio = 0.12
    GSup = np.copy(img)
    h = int(GSup.shape[0])
    w = int(GSup.shape[1])
    highThreshold = np.max(GSup) * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio    
    x = 0.1
    oldx=0

    for i in range(1,h-1):
        for j in range(1,w-1):
            if(GSup[i,j] > highThreshold):
                GSup[i,j] = 1
            elif(GSup[i,j] < lowThreshold):
                GSup[i,j] = 0
            else:
                if((GSup[i-1,j-1] > highThreshold) or 
                    (GSup[i-1,j] > highThreshold) or
                    (GSup[i-1,j+1] > highThreshold) or
                    (GSup[i,j-1] > highThreshold) or
                    (GSup[i,j+1] > highThreshold) or
                    (GSup[i+1,j-1] > highThreshold) or
                    (GSup[i+1,j] > highThreshold) or
                    (GSup[i+1,j+1] > highThreshold)):
                    GSup[i,j] = 1
    
    GSup = (GSup == 1) * GSup 
    
    return GSup

final_img = DoThreshHyst(img_NMS)
plt.imshow(final_img, cmap = plt.get_cmap('gray'))
plt.show()

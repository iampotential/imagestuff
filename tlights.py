
import cv2
import numpy as np
  
# Read image
image = cv2.imread('newhouseimg.png')
# image[:,:,:] = 0
# Convert image to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  
# Use canny edge detection
edges = cv2.Canny(cv2.blur(gray,(5,5)),100,200,apertureSize=3)
  


# image[227:495,255:355,1] = 255
# image[297:405,255:355,0] = 255
from matplotlib import pyplot as plt
plt.imshow(image)
plt.show()
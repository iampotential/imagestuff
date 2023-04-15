
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans


img=cv2.imread('himg.jpg')
shape = img.shape

plt.imshow(img)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

img=img.reshape((img.shape[1]*img.shape[0],3))

kmeans=KMeans(n_clusters=5)
s=kmeans.fit(img)

maps = {0:[0,0,255],1:[100,100,100],2:[255,255,255],3:[0,255,0],4:[255,0,0]}
labels=kmeans.labels_

labels=list(labels)
centroid=kmeans.cluster_centers_
percent=[]
for i in range(len(centroid)):
  j=labels.count(i)
  j=j/(len(labels))
  percent.append(j)
r = s.predict(img)
f = r.reshape(shape[0], shape[1])
print(f.shape)
canny = cv2.Canny(f,100,200)
plt.imshow(canny)
plt.show()
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math
import cv2

import rembg
mul = np.array([[
    [0,1,1,1,1],[0,1,1,1,1],[0,1,1,1,1],[0,1,1,1,1]],
    [
    [0,1,1,1,1],[0,1,1,1,1],[0,1,1,1,1],[0,1,1,1,1]],
    [
    [0,1,1,1,1],[0,1,1,1,1],[0,1,1,1,1],[0,1,1,1,1]]])
mul = mul.reshape(4,5,3)

myimage = Image.open("himg.jpg")
removed = np.array(rembg.remove(myimage))

arr = np.array(myimage)
arrcopy = arr.copy()
arr[removed[:,:,3]>0] = 255
arr[removed[:,:,3]==0] = 0
Image.fromarray(arr).save("newhouseimg.png")

arr = cv2.blur(arr,(4,4))
arr2 = np.array(myimage)
a2 = np.zeros_like(arr)
d = math.gcd(arr.shape[0],arr.shape[1])

avg = arr.mean()


z = mul.shape[0]*4
xx = mul.shape[1]*4

def run_filter(arr,z,xx):
    #step 1
    # create a new mask which hides high quantity of light. First step is to find all the locatoins wiht high light content
    for p in range(0,arr.shape[0],z):
        for k in range(0,arr.shape[1],xx):
     
            try:
                m = int((arr[p:p+z,k:k+xx,:]).mean())
                # if m > avg:
                    
                #     arr[p:p+z,k:k+xx,:] = 255
                    
                if m > avg*2:
                    arr[p:p+z,k:k+xx,:] = 255
                else:
                    arr[p:p+z,k:k+xx,:] = 0
            except:
                print(f"error at {p},{k}")

    return arr 

first_pass = run_filter(arr,z,xx)
cndpass = run_filter(arr,z*4,xx*4)
arr2[cndpass!=0] = 0
# plt.imshow(arr2)
plt.show()

'''h = np.vsplit(arr,d)
newarrs = [np.hsplit(k,d) for k in h]
for item in newarrs:
    reconst.append(np.hstack(item))


final = np.vstack(reconst)
plt.imshow(final)
plt.show()'''
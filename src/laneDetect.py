# Library imports
from statistics import mode
import cv2
import os
import numpy as np

# Grab image in a format that openCV can work with
picpath = os.path.join("Pictures", "test_img.jpg")
origimg = cv2.imread(picpath, cv2.IMREAD_UNCHANGED)

# Apply a gaussian blur to smoothen out the picture. This removes small edges that we aren't really looking for. 
img = cv2.GaussianBlur(origimg,(5,5),cv2.BORDER_DEFAULT)

# Convert to black and white image.
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a threshold. This reduces the amount of work we need to do to find the edges within the image. 
cv2.threshold(img, 150, 255, type=cv2.THRESH_BINARY, dst=img)

# Use a perspective transformation to get a "bird's eye" view of the ground.
height, width = img.shape

# Input points
# Order: Top left, bottom left, bottom right, top right,
A1 = [int(width*0.45),int(height*0.6)]
B1 = [int(width*0.2),int(height*0.9)]
C1 = [int(width*0.75),int(height*0.90)]
D1 = [int(width*0.55),int(height*0.6)]
inpts = [A1,B1,C1,D1]
# Output points 
A2 = [0,0]
B2 = [0, height-1]
C2 = [width-1,height-1]
D2 = [width-1,0]
outpts = [A2,B2,C2,D2]

inpts = np.float32(inpts)
outpts = np.float32(outpts)

matrix = cv2.getPerspectiveTransform(inpts, outpts)

warp = cv2.warpPerspective(img, matrix, (width,height))

contours = []

# Get contours

# Transform each pixel in the contour to the actual image


cv2.imshow('Image',warp)
cv2.waitKey(0)
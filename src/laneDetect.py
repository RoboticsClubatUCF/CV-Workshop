# Library imports
import cv2
import os
import numpy as np

# Grab image in a format that openCV can work with
picPath = os.path.join("Pictures", "test_img3.jpg")
origImg = cv2.imread(picPath, cv2.IMREAD_UNCHANGED)

# Apply a gaussian blur to smoothen out the picture. This removes small edges that we aren't really looking for. 
img = cv2.GaussianBlur(origImg,(5,5),cv2.BORDER_DEFAULT)

# Convert to black and white image.
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a threshold. This reduces the amount of work we need to do to find the edges within the image. 
cv2.threshold(img, 150, 255, type=cv2.THRESH_BINARY, dst=img)

# Use a perspective transformation to get a "bird's eye" view of the ground.
height, width = img.shape

# Input points
# Order: Top left, bottom left, bottom right, top right,
A1 = [int(width*0.45),int(height*0.6)]
B1 = [int(width*0.20),int(height*0.95)]
C1 = [int(width*0.80),int(height*0.92)]
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

# Get edges
edges = cv2.Canny(warp, 100,100)

# Inverse transform 
invTrans = np.linalg.pinv(matrix)
lanes = cv2.warpPerspective(warp, invTrans, (width, height))

# Change image color
lanes = cv2.cvtColor(lanes, cv2.COLOR_GRAY2BGR)

# Turn white lines into green lines 
whiteHi = np.array([255,255,255])
whiteLo = np.array([25,25,25])

# Get white areas in image
mask = cv2.inRange(lanes, whiteLo, whiteHi)

# Replace white with green (blue, green, red) using boolean indexing
lanes[mask>0]=(0,255,0)

# Overlay lanes over original image 
laneDetect = cv2.addWeighted(origImg, 0.6, lanes, 0.4,0)

# Initialize haar cascade 
carCascade = cv2.CascadeClassifier()
paramPath = os.path.join("cascadeParams", "cars.xml")
carCascade.load(paramPath)

# Detect cars
cars = carCascade.detectMultiScale(origImg)

# draw bounding boxes (top left, bottom right)
for (x,y,w,h) in cars:
    cv2.rectangle(laneDetect, (x,y), ((x+w), (y+h)), (0,0,255))

cv2.imshow('Image',laneDetect)
cv2.waitKey(0)
import cv2
import numpy as np
print(cv2.__version__)
# version check 4.5 smth..
# A comment

img = cv2.imread('C:\\Users\\nazar\\Downloads\\ComputerVision\\SBR.jpg') 
img2Face = cv2.imread('C:\\Users\\nazar\\Downloads\\ComputerVision\\myhonestreaction.jpeg') 

# Absolute path.
# For facial recognition us myhonestreaction.jpeg

# Displaying the first image
"""
cv2.imshow('Steel Ball Run Display',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

#displaying the second image

cv2.imshow('Ryan Gosling',img2Face)
cv2.waitKey(0)
cv2.destroyAllWindows()


# detecting edges for the first image
"""
cv2.imwrite('edges_py.jpg',cv2.Canny(img,602,315))
cv2.imshow('edge detection',cv2.imread('edges_py.jpg'))
cv2.waitKey(0) # this function is important for the window pop up
cv2.destroyAllWindows()

print( img.size )
 1218900 is the image size.
 """


# Region of Interest experiment on the second Image
"""
y,x=100,50

(b,g,r)=img2Face[y,x] #Reading color values at y, x positions
print(b,g,r) #Printing color values to screen
# 74 77 82


img2Face[y,x]=(0,0,255) #Setting pixel color to red; BGR scheme
region_of_interest=img2Face[y:y+50,x:x+50] #Region of interest at (x,y) of dimensions 50x50
cv2.imshow('Region of Interest',img2Face)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('ROI',region_of_interest)
region_of_interest[:,:]=(66,22,78) #Setting pixels to new color
cv2.imshow('ROI Red Pixel',img2Face)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# Drawing with Open CV
"""
# Drawing a line + circle + rectangle on Open CV
img1=np.zeros((512,512,3),np.uint8)
# Defining a black background ^^^

cv2.line(img1,(0,0),(511,511),(255,0,0),5)
cv2.circle(img1,(447,63),63,(0,0,255),-1)
cv2.rectangle(img1,(384,0),(510,128),(0,255,0),3)
cv2.imshow('Shapes and Colours!',img1)
# Adding text
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img1,'Hello!',(10,500),font,4,(255,255,23),2,cv2.LINE_AA)
cv2.imshow('Words words words',img1)
cv2.waitKey(0) # this function is important for the window pop up
cv2.destroyAllWindows()
"""

"""
# Facial detection with the second image.
cv2.imwrite('edges_py.jpg',cv2.Canny(img2Face,602,315))
cv2.imshow('edge detection',cv2.imread('edges_py.jpg'))
cv2.waitKey(0) # this function is important for the window pop up
cv2.destroyAllWindows()
"""

## Face Detection and Eye Detection with Image 2
fd=cv2.CascadeClassifier('C:\\Users\\nazar\\Downloads\\ComputerVision\\haarcascade_frontalface_default.xml')
gray=cv2.cvtColor(img2Face,cv2.COLOR_BGR2GRAY) #Converting it to grayscale
faces=fd.detectMultiScale(gray,1.3,5) #Performing the detection

for (x,y,w,h) in faces:
    img2Face=cv2.rectangle(img2Face,(x,y),(x+w,y+h),(255,0,0),3)

cv2.imwrite('face_Gosling.jpg',img2Face) # note: the images are written in the file.
cv2.waitKey(0) # this function is important for the window pop up
cv2.destroyAllWindows()

eye_cascade = cv2.CascadeClassifier('C:\\Users\\nazar\\Downloads\\ComputerVision\\haarcascade_eye.xml')
# Draw a rectangle around the faces
roi_gray = gray[y:y+h, x:x+w]
roi_color = img2Face[y:y+h, x:x+w]
eyes = eye_cascade.detectMultiScale(roi_gray)
for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imwrite("Eyes_found.jpg" ,img2Face)
cv2.waitKey(0)

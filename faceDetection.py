import cv2

faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
img = cv2.imread("Resources/vsj.jpg") #just copy the image path here
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #converting image to gray scale

faces = faceCascade.detectMultiScale(imgGray, 1.1, 13)

for(x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

cv2.imshow("Output", img)
cv2.waitKey(0)
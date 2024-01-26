###########################################################################
#                        Face Detection using DLIB                        #
###########################################################################

import numpy as np
import dlib,cv2

def dlib_bounding_to_rect(rect_boxes):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect_boxes.left()
    y = rect_boxes.top()
    w = rect_boxes.right() - x
    h = rect_boxes.bottom() - y
    return (x, y, w, h)

# initialize dlib's face detector (HOG-based) 
detector = dlib.get_frontal_face_detector()

# load the input image, resize it, and convert it to grayscale
imageFilename = "ellen_oscar_selfie.jpg"
#imageFilename = "dog_human.jpg"
image = cv2.imread(imageFilename)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

facesDetected = detector(gray, 1)
print("Number of faces detected: ",len(facesDetected))
print(facesDetected)


# Loop over all detected face rectangles
for faces in facesDetected: 
  (x, y, w, h) = dlib_bounding_to_rect(faces)
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
cv2.imshow("Face Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows() 
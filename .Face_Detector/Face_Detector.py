# Import the necessary libraries
import cv2
from random import randrange

# Load a bunch of pre-trained face data
trained_face_data = cv2.CascadeClassifier(
    r"C:\Users\paom1\PycharmProjects\FirstAI\.Face_Detector\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

# Load an image
img = cv2.imread(r"Faces.jpg")


# Covert it all to Greyscale
grayscaled_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

''' At this point, we would train the algorithm to detect faces, but open CV has already provided us with a pre-trained algorithm that we can use, thus we can just move on to the next phase'''

# Detect the face, this gives us the coordinates of the rectangle that highlights the face
face_coordiantes = trained_face_data.detectMultiScale(grayscaled_image)
print(face_coordiantes)

# Draw the rectangle around the face
# this 4-touple will assign the each of the output numbers to a different name variable
for (x, y, w, h) in face_coordiantes:
# then we can input those variables into the rectangle drawing function
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 5)

# Show us the image so we can confirm its working
cv2.imshow('Clever Programmer Face Detector', img)
cv2.waitKey()

# Code Execution Confirmation

people_count = len(face_coordiantes)
print(f'The program has detected {people_count} people in this image')

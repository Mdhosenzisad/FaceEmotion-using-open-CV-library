import cv2
import numpy as np

# Load the face detection and emotion recognition models
from facial_emotion_recognition import EmotionRecognition

er = EmotionRecognition(device="cpu")
cap = cv2.VideoCapture(0)


while True:
    success, frame = cap.read()
    frame = er.recognise_emotion(frame, return_type='BGR')
    cv2.imshow('frame', frame)

    # exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()


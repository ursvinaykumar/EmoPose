

import cv2
import mediapipe as mp
from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load emotion detection model and class labels
classifier = load_model('C:/Users/VINAY KUMAR/Desktop/project/Emotion/Emotion_Detection.h5')
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load Haar cascade classifier for face detection
face_classifier = cv2.CascadeClassifier('C:/Users/VINAY KUMAR/Desktop/project/Emotion/haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Pose Detection
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detection:
    while True:
        # Read each frame from the webcam
        ret, image = cap.read()
        if not ret:
            break

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect the pose in the image
        results = pose_detection.process(image_rgb)

        # Draw the pose on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Perform emotion detection on faces in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Make a prediction on the ROI, then lookup the class
                preds = classifier.predict(roi)[0]
                label = class_labels[preds.argmax()]
                label_position = (x, y)
                cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(image, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # Show the image with pose and emotion detection overlay
        cv2.imshow('Pose and Emotion Detection', image)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()

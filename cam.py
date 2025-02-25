import cv2
import numpy as np
import tensorflow as tf
import pygame  


pygame.mixer.init()

try:
    model = tf.keras.models.load_model('new_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty() or eye_cascade.empty():
    print("Error: Could not load Haar cascades.")
    exit()

def get_model_prediction(face_roi):
    try:
        face_resized = cv2.resize(face_roi, (128, 128))
        face_normalized = face_resized / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=0)
        prediction = model.predict(face_expanded)
        return prediction[0][0]  
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

try:
    sound = pygame.mixer.Sound('sound1.mp3')  
except pygame.error as e:
    print(f"Error loading sound file: {e}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        prediction = get_model_prediction(roi_color)
        if prediction is not None:
            print(f"Prediction: {prediction}")
            if prediction < 0.5:  
                sound.play()
            else:
                sound.stop() 

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit() 

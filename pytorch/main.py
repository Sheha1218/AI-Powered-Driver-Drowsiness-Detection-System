from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

import torch
import cv2 as cv
import numpy as np
import pygame
import time

import mediapipe as mp
from scipy.spatial import distance as dist

from pytorch.detector import detect


app = FastAPI()
templates = Jinja2Templates(directory="frontend")


class_names = ['Closed_Eyes', 'Open_Eyes']

classifier = detect(
    r"D:\Way to Denmark\Projects\AI-Powered-Driver-Drowsiness-Detection-System\pytorch\driver.pth",
    class_names
)

pygame.mixer.init()
alert_sound = pygame.mixer.Sound(r"pytorch\sound1.mp3")


cap = cv.VideoCapture(0)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESH = 0.20
EAR_CONSEC_FRAMES = 2
ear_counter = 0

# Alert control
last_alert_time = 0
ALERT_DELAY = 2  # seconds



def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def preprocess(frame):
    frame_resized = cv.resize(frame, (128, 128))
    frame_lab = cv.cvtColor(frame_resized, cv.COLOR_BGR2LAB)
    frame_lab = frame_lab.astype("float32") / 255.0
    frame_transposed = np.transpose(frame_lab, (2, 0, 1))
    frame_expanded = np.expand_dims(frame_transposed, axis=0)
    return torch.tensor(frame_expanded)



def gen_frames():
    global ear_counter, last_alert_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

       
        input_tensor = preprocess(frame)
        label, confidence = classifier.predict(input_tensor)

        
        results = face_mesh.process(rgb)
        ear = None

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = []
                right_eye = []

                for idx in LEFT_EYE:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    left_eye.append((x, y))

                for idx in RIGHT_EYE:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    right_eye.append((x, y))

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

       
        drowsy = False

        if ear is not None and ear < EAR_THRESH:
            ear_counter += 1
        else:
            ear_counter = 0

        if ear_counter >= EAR_CONSEC_FRAMES or label == "Closed_Eyes":
            drowsy = True

        
        if drowsy:
            current_time = time.time()
            if current_time - last_alert_time > ALERT_DELAY:
                alert_sound.play()
                last_alert_time = current_time

       
        cv.rectangle(frame, (0, 0), (frame.shape[1], 80), (30, 30, 30), -1)

        cv.putText(frame, f"Model: {label} ({confidence:.2f})",
                   (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if ear is not None:
            cv.putText(frame, f"EAR: {ear:.2f}",
                       (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if drowsy:
            cv.putText(frame, "DROWSINESS DETECTED",
                       (350, 55), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 255), 3)

       
        _, buffer = cv.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )



@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.get("/video")
def video():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

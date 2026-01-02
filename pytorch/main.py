import torch
import cv2
from torchvision import transforms
from detector import detect
import pygame
import time



face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)



pygame.init()
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("pytorch/sound1.mp3")

last_alert_time = 0
alert_cooldown = 3

def play_alert():
    global last_alert_time
    now = time.time()
    if now - last_alert_time > alert_cooldown:
        alert_sound.play()
        last_alert_time = now



classifier = detect(
    model_path=r"pytorch\driver.pth",
    class_names=['Closed_Eyes','Open_Eyes']
)

if not hasattr(classifier, "model"):
    print("Model not loaded")
    exit()

print("Model loaded")



transform = transforms.Compose([
    transforms.ToTensor()
])



cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(" Camera doesnt open")
    exit()


closed_frames = 0
drowsy_threshold = 20



while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]
            eye_img = cv2.resize(eye_img, (128, 128))

            tensor = transform(eye_img).unsqueeze(0)

            try:
                label, conf = classifier.predict(tensor)
            except Exception as e:
                print("Prediction failed:", e)
                continue

            color = (0,0,255) if label == "Closed_Eyes" else (0,255,0)

            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex+ew, ey+eh),
                color, 2
            )

            cv2.putText(
                roi_color,
                f"{label} {conf:.2f}",
                (ex, ey-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

            if label == "Closed_Eyes":
                closed_frames +=1
            else:
                closed_frames=0

            if closed_frames >= drowsy_threshold:
                cv2.putText(
                    frame,
                    "Drowsyness altert",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0,0,255),
                    2
                )
            else:
                
                play_alert()

    cv2.imshow("Driver drowsiness detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
pygame.quit()

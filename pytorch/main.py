import torch
import cv2
from torchvision import transforms
from detector import detect
import pygame
import time


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


pygame.init()
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('pytorch/sound1.mp3')  # your alert sound

# Cooldown for sound alert (seconds)
last_alert_time = 0
alert_cooldown = 3

def play_alert():
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time > alert_cooldown:
        alert_sound.play()
        last_alert_time = current_time


classifier = detect(
    model_path=r'pytorch\driver.pth',
    class_names=['Closed_Eyes', 'Open_Eyes']
)

if hasattr(classifier, 'model'):
    print("Model loaded successfully")
else:
    print("Model did not load")
    exit()


transform = transforms.Compose([transforms.ToTensor()])


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera dosnt open")
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
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (128, 128))
        tensor = transform(face_img).unsqueeze(0)

        
        try:
            label, conf = classifier.predict(tensor)
        except Exception as e:
            print("Prediction failed:", e)
            continue

        
        color = (0, 255, 0) if label == 'Closed_Eyes' else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

       
        if label == 'Closed_Eyes':
            closed_frames += 1
        else:
            closed_frames = 0

        
        if closed_frames >= drowsy_threshold:
            cv2.putText(frame, "Drowsiness", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            play_alert()

  
    cv2.imshow("Driver drowsiness detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
pygame.quit()

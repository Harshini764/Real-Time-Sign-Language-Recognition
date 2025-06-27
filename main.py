import cv2
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model
from collections import deque

# Load trained model
model = load_model("model/isl_model.h5")

# Labels: 1–9 + A–Z
labels = list("123456789") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Prediction smoothing buffer
pred_buffer = deque(maxlen=15)
last_spoken = None

# Start webcam
cap = cv2.VideoCapture(0)
print("📸 Webcam started. Show a gesture inside the box.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define ROI (region of interest)
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)

    # Preprocess ROI for model
    img = cv2.resize(roi, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img)
    confidence = np.max(predictions)
    pred_index = np.argmax(predictions)
    pred_letter = labels[pred_index]

    # Smooth prediction
    pred_buffer.append(pred_letter)
    stable_prediction = None
    if len(pred_buffer) == pred_buffer.maxlen:
        if len(set(pred_buffer)) == 1 and confidence > 0.8:
            stable_prediction = pred_letter

    # Display confidence
    if confidence > 0.8:
        cv2.putText(frame, f"{pred_letter} ({confidence*100:.1f}%)", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "Not recognized", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Speak once if new prediction is stable
    if stable_prediction and stable_prediction != last_spoken:
        speak(stable_prediction)
        last_spoken = stable_prediction

    # Show frame
    cv2.imshow("ISL to Speech", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

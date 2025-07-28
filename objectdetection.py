import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pyttsx3
import threading
import time

# Initialize YOLO model
model = YOLO("yolov8n.pt")  # Replace with your custom model if needed

# Initialize Streamlit
st.title("Real-time Obstacle Detection")


# Initialize session state to track spoken status
if "already_spoken" not in st.session_state:
    st.session_state["already_spoken"] = False

# Thread-safe TTS function
def speak(text):
    def run_tts():
        local_engine = pyttsx3.init()
        local_engine.setProperty('rate', 150)
        local_engine.say(text)
        local_engine.runAndWait()
    threading.Thread(target=run_tts).start()

# Start webcam and detection
start = st.button("Start Detection")
stop = st.button("Stop Detection")

if start and not stop:
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame from camera.")
            break

        # Object detection
        results = model(frame)

        # Draw bounding boxes and check for persons
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            label = results[0].names[int(class_id)]
            if label == 'person' and score > 0.5:
                # Voice alert once every few seconds
                if not st.session_state["already_spoken"]:
                    speak("Warning, there is someone in front of you")
                    st.session_state["already_spoken"] = True

                # Draw box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10) ,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Convert and display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Reset speaking flag every 5 seconds
        if time.time() % 5 < 0.3:
            st.session_state["already_spoken"] = False

        # Stop logic
        if stop:
            break

    cap.release()
    cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import threading  

# Initialize MediaPipe Pose and TTS
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
engine = pyttsx3.init()

# Set voice properties
engine.setProperty("rate", 170)  # Adjust speech speed if needed
engine.setProperty("volume", 1.0)  # Max volume

# Fix pronunciation by replacing "push-ups" with "push-up"
def speak(text):
    """Function to convert text to speech in a separate thread."""
    text = text.replace("push-ups", "push-up")  # Say "push-up" instead

    def run_speech():
        engine.say(text)
        engine.runAndWait()
    
    threading.Thread(target=run_speech, daemon=True).start()

# Start video capture
cap = cv2.VideoCapture(0)

# Push-up tracking variables
pushup_count = 0
pushup_state = "Up"
last_pushup_time = time.time()  # Track last push-up time
inactive_time_threshold = 10  # 10 seconds of inactivity

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        h, w, _ = frame.shape
        shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
        elbow_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * h
        wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * h

        # Check push-up position
        if elbow_y > shoulder_y and wrist_y > elbow_y:  # Down position
            if pushup_state == "Up":
                pushup_state = "Down"

        elif elbow_y < shoulder_y:  # Up position
            if pushup_state == "Down":
                pushup_state = "Up"
                pushup_count += 1  # Count rep when coming up
                last_pushup_time = time.time()  # Reset inactivity timer

                # Announce only at multiples of 10
                if pushup_count % 10 == 0:
                    speak(f"Congratulations! You have completed {pushup_count} push-up!")

        # Display count on screen
        cv2.putText(frame, f"Push-up: {pushup_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Check inactivity
    if time.time() - last_pushup_time > inactive_time_threshold:
        speak(f"Come on! No push-up detected for 10 seconds. You completed {pushup_count} push-up. Keep going!")
        last_pushup_time = time.time()  # Reset inactivity timer

    cv2.imshow('Push-up Counter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
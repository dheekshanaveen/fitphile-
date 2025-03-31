import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time

# Initialize MediaPipe Pose & TTS
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
engine = pyttsx3.init()

# Set voice properties
engine.setProperty("rate", 170)  # Speech speed
engine.setProperty("volume", 1.0)  # Max volume

# Function to announce messages
def speak(text):
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech, daemon=True).start()

# Start video capture
cap = cv2.VideoCapture(0)

# Squat tracking variables
squat_count = 0
in_squat = False
last_announcement_count = 0  # Prevent repeated announcements
last_squat_time = time.time()  # Track last squat time
motivation_given = False  # Track if motivation message is given

def calculate_angle(a, b, c):
    """Calculate angle between three points using cosine rule"""
    a = np.array(a)  
    b = np.array(b)  
    c = np.array(c)  
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    h, w, _ = frame.shape

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get key body part positions
        landmarks = results.pose_landmarks.landmark

        hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
        knee = (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * h))
        ankle = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h))

        left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
        left_knee = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * h))
        left_ankle = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h))

        # Calculate knee angles
        right_knee_angle = calculate_angle(hip, knee, ankle)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

        # Show angle on screen
        cv2.putText(frame, f"Right Angle: {int(right_knee_angle)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Left Angle: {int(left_knee_angle)}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        # Squat detection logic
        if right_knee_angle < 90 and left_knee_angle < 90:
            in_squat = True  # Person is in squat position
        elif right_knee_angle > 160 and left_knee_angle > 160 and in_squat:
            squat_count += 1  # Count squat when standing back up
            in_squat = False
            last_squat_time = time.time()  # Reset timer on squat
            motivation_given = False  # Reset motivation flag

        # Intensity logic (Depth-based)
        intensity = max(0, min(100, int((160 - min(right_knee_angle, left_knee_angle)) / 70 * 100)))

        # Draw intensity bar
        cv2.rectangle(frame, (30, 50), (70, 400), (50, 50, 50), 2)  # Bar border
        bar_height = int((intensity / 100) * 350)
        cv2.rectangle(frame, (30, 400 - bar_height), (70, 400), (0, 255, 255), -1)  # Fill intensity bar
        cv2.putText(frame, f"{intensity}%", (30, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Announce after every 10 squats (only once per 10 reps)
        if squat_count % 10 == 0 and squat_count > 0 and squat_count != last_announcement_count:
            speak(f"Congratulations! You have completed {squat_count} squats!")
            last_announcement_count = squat_count  # Prevent repeated announcements

        # *Motivation check*: If no squat for 10 seconds, give motivation (only once per inactivity period)
        if time.time() - last_squat_time > 10 and not motivation_given:
            speak("Come on! Keep pushing! Squats are your power!")
            motivation_given = True  # Ensure message is not repeated continuously

        # Display count
        cv2.putText(frame, f"Squats: {squat_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Squat Tracker with Angles & Intensity', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
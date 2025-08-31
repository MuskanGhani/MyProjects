import cv2
import mediapipe as mp
import time
import datetime
import math

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Initialize camera
cap = cv2.VideoCapture(0)

smile_count = 0
cooldown = 0

print("😊 Smile detection started... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmark coordinates
            def get_point(id):
                lm = face_landmarks.landmark[id]
                return int(lm.x * w), int(lm.y * h)

            left = get_point(61)
            right = get_point(291)
            top = get_point(13)
            bottom = get_point(14)

            # Calculate mouth width and height
            mouth_width = math.dist(left, right)
            mouth_height = math.dist(top, bottom)

            smile_ratio = mouth_width / (mouth_height + 1e-6)  # avoid div by zero

            # Draw points
            cv2.circle(frame, left, 2, (0, 255, 0), -1)
            cv2.circle(frame, right, 2, (0, 255, 0), -1)

            if smile_ratio > 3.5 and time.time() > cooldown:  # Threshold to detect smile
                smile_count += 1
                filename = f"smile_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[{smile_count}] Smile captured: {filename}")
                cooldown = time.time() + 2

            # Optional: draw face mesh
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1))

    cv2.imshow("MediaPipe Smile Detector - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Smile detection ended.")

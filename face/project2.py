import face_recognition
import cv2
import os
import csv
from datetime import datetime

# --- Setup ---
db_path = "known_faces"
attendance_file = "attendance.csv"
marked_today = set()

# --- Create attendance.csv if not exists ---
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

# --- Mark attendance ---
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    if name not in marked_today:#avoids duplicates
        with open(attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date, time])
        marked_today.add(name)
        print(f"Marked: {name} at {time}")

# --- Load known faces ---
known_encodings = []
known_names = []

for filename in os.listdir(db_path):
    if filename.endswith(('.jpg', '.png')):
        path = os.path.join(db_path, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)# converts the face into a 128-dimensional vector (encoding).

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])

# --- Start webcam ---
video = cv2.VideoCapture(0)

print("Starting attendance... Press Q to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
            mark_attendance(name)

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import os
import time
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import DetrImageProcessor, DetrForSegmentation

# ----------------- CNN for lipstick classification -----------------
class LipstickDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        label = np.argmax([row['cool'], row['neutral'], row['warm']])
        if self.transform:
            image = self.transform(image)
        return image, label

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training CNN if model not already saved
model_path = "lipstick_classifier.pth"
if not os.path.exists(model_path):
    print("Training lipstick CNN model...")
    dataset = LipstickDataset("C:/Users/HP/PycharmProjects/PythonProject2/lip_theory/train/_classes.csv",
                              "C:/Users/HP/PycharmProjects/PythonProject2/lip_theory/train", transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    cnn_model = SimpleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
    for epoch in range(5):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = cnn_model(images)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(cnn_model, model_path)
    print("Model trained and saved.")
else:
    cnn_model = torch.load(model_path).to(device)
cnn_model.eval()

# ----------------- MediaPipe + Real-time Detection -----------------
df = pd.read_csv(r"C:/Users/HP/PycharmProjects/PythonProject2/lip_theory/train/_classes.csv")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)

# Initialize DETR
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic").to(device)

if not cap.isOpened():
    print("Error: Webcam not found!")
    exit()

start_time = time.time()
fixed_lipstick = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Lip landmarks for both upper and lower lips
            lip_points = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375,
                          321, 405, 17, 84, 91, 146, 61, 78, 95, 88, 178, 87,
                          14, 317, 402, 318, 324, 308]
            lip_coords = np.array([
                [int(face_landmarks.landmark[p].x * frame.shape[1]),
                 int(face_landmarks.landmark[p].y * frame.shape[0])]
                for p in lip_points
            ])
            mask = np.zeros_like(frame, dtype=np.uint8)
            cv2.fillPoly(mask, [lip_coords], (255, 255, 255))

            if fixed_lipstick is None and time.time() - start_time < 3:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = processor(images=pil_img, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[pil_img.size])[0]
                seg_mask = result['segmentation'].cpu().numpy()
                segments = result['segments_info']
                person_mask = None
                for segment in segments:
                    if segment['label_id'] == 1:
                        person_mask = (seg_mask == segment['id']).astype(np.uint8) * 255
                        person_mask = cv2.resize(person_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                        break

                if person_mask is not None:
                    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                    skin_ycrcb = ycrcb[person_mask == 255]
                    avg_cr = np.mean(skin_ycrcb[:, 1])
                    avg_cb = np.mean(skin_ycrcb[:, 2])
                    if avg_cr > avg_cb + 10:
                        undertone = "warm"
                    elif avg_cb > avg_cr + 10:
                        undertone = "cool"
                    else:
                        undertone = "neutral"
                    print("User Undertone:", undertone)

                    filtered_df = df[df[undertone] == 1]
                    if filtered_df.empty:
                        print(f"No matching lipstick for {undertone}.")
                        break

                    lipstick_choice = filtered_df.sample(1)["filename"].values[0]
                    lipstick_color = cv2.imread(f"C:/Users/HP/PycharmProjects/PythonProject2/lip_theory/train/{lipstick_choice}")
                    if lipstick_color is None:
                        print("Error: Lipstick image not found.")
                        break

                    # 🔍 Predict lipstick tone using CNN
                    pil = Image.fromarray(cv2.cvtColor(lipstick_color, cv2.COLOR_BGR2RGB))
                    tensor_img = transform(pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred = cnn_model(tensor_img)
                    predicted = torch.argmax(pred).item()
                    pred_label = [""][predicted]
                    print("Lipstick Predicted Tone:", pred_label)

                    if pred_label != undertone:
                        print("⚠️ Lipstick undertone doesn't match user undertone!")

                    fixed_lipstick = cv2.resize(lipstick_color, (frame.shape[1], frame.shape[0]))

            if fixed_lipstick is not None:
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask_blur = cv2.GaussianBlur(mask_gray, (15, 15), 10)
                mask_blur = cv2.cvtColor(mask_blur, cv2.COLOR_GRAY2BGR) / 255.0
                # Smoothing the blend and adding color adjustment for more natural effect
                blended = (frame * (1 - mask_blur) + fixed_lipstick * mask_blur).astype(np.uint8)
                blended = cv2.addWeighted(blended, 0.9, frame, 0.1, 0)  # Adjust transparency for smoothness

                # Display message on screen
                cv2.putText(blended, f"Undertone: {undertone}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Lipstick Applied", blended)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import os
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Dropout, TimeDistributed
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Step 1: Load YOLOv8 Model
model_yolo = YOLO('yolov8n.pt')  # Load pre-trained YOLOv8 model

# Step 2: Initialize MediaPipe for Keypoint Extraction
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Step 3: Data Preprocessing
def extract_keypoints(video_path, yolo_model):
    """Extract keypoints using MediaPipe and YOLO from a video."""
    cap = cv2.VideoCapture(video_path)
    holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    keypoints_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Use YOLO to detect hands
        results_yolo = yolo_model(frame, verbose=False)
        detections= results_yolo[0].boxes.data.cpu().numpy()
        for box in detections:  # Loop over detected objects
            x1, y1, x2, y2, conf, cls = box
            if int(cls) == 0:  # Assuming class 0 is 'hand'
                hand_roi = frame[int(y1):int(y2), int(x1):int(x2)]

        # Extract keypoints using MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        if results.left_hand_landmarks:
            keypoints = []
            for lm in results.left_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            keypoints_sequence.append(keypoints)
        else:
            keypoints_sequence.append([0] * 63)  # No keypoints detected, pad with zeros

    cap.release()
    holistic.close()
    return np.array(keypoints_sequence)

# Step 4: Prepare Dataset
def process_video_folders(folder_path, yolo_model):
    """Process all videos in folders and create dataset."""
    data = []
    label_list = []
    labels_dict = {label: idx for idx, label in enumerate(os.listdir(folder_path))}

    for label, idx in labels_dict.items():
        label_folder = os.path.join(folder_path, label)
        for video_file in os.listdir(label_folder):
            video_path = os.path.join(label_folder, video_file)
            keypoints_sequence = extract_keypoints(video_path, yolo_model)
            data.append(keypoints_sequence)
            label_list.append(idx)

    return np.array(data), np.array(label_list), labels_dict

# Example Dataset Path
dataset_folder = "Greetings"  # Path to the folder containing subfolders for each word

# Process dataset
data, labels, labels_dict = process_video_folders(dataset_folder, model_yolo)
data = np.array([np.pad(seq, ((0, 100 - len(seq)), (0, 0)), mode='constant') for seq in data])  # Pad sequences to length 100
labels = to_categorical(labels)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Step 5: Define CNN+LSTM Model
def build_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(100, 63, 1)))  # Adjust dimensions if using raw frames
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(labels.shape[1], activation='softmax'))
    return model

# Build and Compile Model
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train the Model
history = model.fit(X_train, y_train, epochs=25, batch_size=8, validation_data=(X_test, y_test))

# Step 7: Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

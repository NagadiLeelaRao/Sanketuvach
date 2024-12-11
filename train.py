import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from data_preparation import process_videos_to_dataset, normalize_landmarks, augment_landmarks, prepare_labels

'''# Preprocessing and Augmentation Functions
def normalize_and_flatten_landmarks(data):
    """
    Normalize landmark data to the range [0, 1].
    """
    data=data / np.max(data, axis=(1, 2), keepdims=True)
    num_landmarks = data.shape[2] * data.shape[3]  # Flatten landmarks and channels
    return data.reshape(data.shape[0], data.shape[1], num_landmarks)

def augment_landmarks(data):
    """
    Augment landmarks by adding noise or transformations.
    """
    augmented = []
    for sample in data:
        noise = np.random.normal(0, 0.01, sample.shape)
        augmented.append(sample + noise)  # Add Gaussian noise
        augmented.append(sample[::-1])   # Reverse sequence
        augmented.append(sample)         # Original
    return np.array(augmented)

def process_videos_to_dataset(video_dir, max_frames=30):
    """
    Process videos from a directory into normalized landmark data and labels.
    """
    X_data = []
    y_data = []
    classes = sorted(os.listdir(video_dir))  # Assume folders are named by class labels
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(video_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for video_file in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_file)
            landmarks = extract_landmarks_from_video(video_path, max_frames)
            if landmarks is not None:
                X_data.append(landmarks)
                y_data.append(label)
    return np.array(X_data), np.array(y_data)

def extract_landmarks_from_video(video_path, max_frames):
    """
    Placeholder: Extract landmarks from video and return as a NumPy array.
    Replace this function with the actual landmark extraction logic.
    """
    # Mock data for demonstration
    num_landmarks = 21  # Number of landmarks
    return np.random.rand(max_frames, num_landmarks, 3)  # (frames, landmarks, x/y/z)

def prepare_labels(labels, num_classes):
    """
    One-hot encode labels.
    """
    return to_categorical(labels, num_classes=num_classes)

# Build Model Function
def build_model(input_shape, num_classes):
    """
    Build a CNN-LSTM model for video classification.
    """
    model = Sequential([
        LSTM(128, return_sequences=True, activation='relu', kernel_regularizer=l2(0.001), input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main Function
def main():
    video_dir = 'Greetings'  # Replace with your video directory path
    model_save_path = 'isl_model.keras'
    max_frames = 30
    batch_size = 16
    epochs = 50

    print("Loading and preprocessing data...")
    X_data, y_data = process_videos_to_dataset(video_dir, max_frames)

    X_data = normalize_and_flatten_landmarks(X_data)

    # Augment data
    X_data_augmented = augment_landmarks(X_data)
    y_data_augmented = np.repeat(y_data, 3, axis=0)  # Repeat labels for each augmented sample

    # One-hot encode labels
    num_classes = len(set(y_data))
    y_data_augmented = prepare_labels(y_data_augmented, num_classes)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_data_augmented, y_data_augmented, test_size=0.2, random_state=42
    )

    print("Building model...")
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    model = build_model(input_shape, num_classes)

    # Callbacks for saving the model and early stopping
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)

    # Compute class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(np.argmax(y_train, axis=1)), 
        y=np.argmax(y_train, axis=1)
    )
    class_weights = dict(enumerate(class_weights))

    print("Starting training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weights
    )

    print(f"Training complete. Model saved to {model_save_path}.")
    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.2f}")

    print("Generating classification report...")
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()'''

def build_model(input_shape, num_classes):
    """
    Build a CNN-LSTM model for video classification.
    """
    model = Sequential([
        LSTM(128, return_sequences=True, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    video_dir = 'Greetings'  # Replace with your video directory path
    model_save_path = 'isl_model.keras'
    max_frames = 30
    #num_classes = 9
    batch_size = 16
    epochs = 50

    print("Loading and preprocessing data...")
    X_data, y_data = process_videos_to_dataset(video_dir, max_frames)

    X_data = normalize_landmarks(X_data)

    # Augment data
    X_data_augmented = augment_landmarks(X_data)
    y_data_augmented = np.repeat(y_data, 3, axis=0)  # Repeat labels for each augmented sample

    # One-hot encode labels
    num_classes = len(set(y_data))
    y_data_augmented= prepare_labels(y_data_augmented, num_classes)

    # Split into train and test sets
    from sklearn.model_selection import train_test_split
    #print(f"X_data shape: {X_data.shape}, y_data shape: {len(y_data)}")
    X_train, X_test, y_train, y_test = train_test_split(X_data_augmented, y_data_augmented, test_size=0.2, random_state=42)
    #print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    #print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    print("Building model...")
    input_shape = (X_train.shape[1], X_train.shape[2]) #X_train.shape[1:]  # (timesteps, features)
    model = build_model(input_shape, num_classes)

    # Callbacks for saving the model and early stopping
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # Use more advanced callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=3, 
        min_lr=0.00001
    )
    
    # Add class weights if slightly imbalanced
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(np.argmax(y_train, axis=1)), 
        y=np.argmax(y_train, axis=1)
    )
    class_weights = dict(enumerate(class_weights))

    print("Starting training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weights
    )

    print(f"Training complete. Model saved to {model_save_path}.")
    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.2f}")

if __name__ == "__main__":
    main()



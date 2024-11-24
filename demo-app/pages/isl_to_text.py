import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import copy
import itertools
import string
import pandas as pd

class SignLanguageApp:
    def __init__(self):
        # MediaPipe hands setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        try:
            self.hands = self.mp_hands.Hands(
                model_complexity=0,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            st.error(f"MediaPipe initialization error: {e}")
        
        # Load ML model
        try:
            self.model = tf.keras.models.load_model(r"/home/avinash/Documents/Sanketuvach/demo-app/models/model.h5") # set absolute path of the model
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
        
        # Define alphabet for predictions
        self.alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)
        
        # Translation dictionary
        self.translations = {
            'English': {},
            'Hindi': {},
            'Marathi': {},
            'Gujarati': {},
            'Bengali': {},
            'Tamil': {},
            'Telugu': {},
            'Kannada': {},
            'Malayalam': {}
        }
        
        # Initialize translations for each letter/number
        for char in self.alphabet:
            for lang in self.translations.keys():
                self.translations[lang][char] = char  # You can replace with actual translations

    def calc_landmark_list(self, image, landmarks):
        """Calculate landmark coordinates"""
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        
        for landmark in landmarks.landmark:
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
            
        return landmark_point

    def pre_process_landmark(self, landmark_list):
        """Preprocess landmarks for model input"""
        temp_landmark_list = copy.deepcopy(landmark_list)
        
        # Convert to relative coordinates
        base_x, base_y = temp_landmark_list[0]
        for point in temp_landmark_list:
            point[0] = point[0] - base_x
            point[1] = point[1] - base_y
        
        # Convert to one-dimensional list
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        
        # Normalization
        max_value = max(map(abs, temp_landmark_list))
        temp_landmark_list = [n / max_value for n in temp_landmark_list]
        
        return temp_landmark_list

    def process_frame(self, frame, selected_language):
        """Process a single frame and return the processed frame and prediction"""
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        prediction = None
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                results.multi_handedness):
                # Calculate and preprocess landmarks
                landmark_list = self.calc_landmark_list(frame, hand_landmarks)
                preprocessed_landmarks = self.pre_process_landmark(landmark_list)
                
                # Prepare data for prediction
                df = pd.DataFrame(preprocessed_landmarks).transpose()
                
                # Make prediction
                predictions = self.model.predict(df, verbose=0)
                predicted_class = np.argmax(predictions, axis=1)[0]
                predicted_char = self.alphabet[predicted_class]
                
                # Get translation
                prediction = self.translations[selected_language].get(predicted_char, predicted_char)
                
                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Add prediction text to frame
                cv2.putText(
                    frame,
                    f"Predicted: {predicted_char}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), prediction

def initialize_camera():
    """Initialize the camera capture"""
    return cv2.VideoCapture(0)

def release_camera():
    """Safely release the camera"""
    if 'camera' in st.session_state and st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
    cv2.destroyAllWindows()

def main():
    st.title("ISL to Text Converter")
    
    # Initialize the app
    app = SignLanguageApp()
    
    # Initialize session state
    if 'camera' not in st.session_state:
        st.session_state.camera = None
    
    # Language selection
    selected_language = st.selectbox(
        "Select Language",
        options=list(app.translations.keys()),
        index=0,
        key='language_select'
    )
    
    # Sidebar instructions
    st.sidebar.header("Instructions")
    st.sidebar.info("""
    1. Click 'Start Camera'
    2. Show hand signs (letters A-Z or numbers 1-9)
    3. Translation will appear below video
    4. Select your preferred language from the dropdown
    """)

    # Camera control
    if st.checkbox('Start Camera', key='camera_checkbox'):
        if st.session_state.camera is None:
            st.session_state.camera = initialize_camera()
        
        # Create placeholders
        frame_window = st.empty()
        translation_container = st.empty()
        
        if st.session_state.camera and st.session_state.camera.isOpened():
            while True:
                ret, frame = st.session_state.camera.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Process frame and get prediction
                processed_frame, prediction = app.process_frame(frame, selected_language)
                
                # Display frame
                frame_window.image(processed_frame)
                
                # Display prediction
                if prediction:
                    translation_container.markdown(f"### Translation: {prediction}")
                else:
                    translation_container.empty()
                
                # Break if checkbox is unchecked
                if not st.session_state.camera_checkbox:
                    break
        else:
            st.error("Cannot open camera. Check connection and permissions.")
    else:
        # Release camera when checkbox is unchecked
        release_camera()

    # Back button
    if st.button("Back to Home"):
        release_camera()
        st.switch_page("Home.py")

if __name__ == "__main__":
    try:
        main()
    finally:
        release_camera()
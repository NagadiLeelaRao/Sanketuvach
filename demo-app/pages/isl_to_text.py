import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

st.set_page_config(
    page_title="Sanketuvach Demo: ISL to Text"
)

class SignLanguageApp:
    def __init__(self):
        # MediaPipe hands setup
        self.mp_hands = mp.solutions.hands
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7
            )
        except Exception as e:
            st.error(f"MediaPipe initialization error: {e}")
        
        # Translation dictionary for Indian languages [TO BE REPLACED BY A TRANSLATION API]
        self.translations = {
            'English': 'Hand Raised',
            'Hindi': 'हाथ उठा',
            'Marathi': 'हात वर',
            'Gujarati': 'હાથ ઊંચો',
            'Bengali': 'হাত তোলা',
            'Tamil': 'கை உயர்த்தப்பட்டது',
            'Telugu': 'చేయి పైకి లేపారు',
            'Kannada': 'ಕೈ ಎತ್ತಿದೆ',
            'Malayalam': 'കൈ ഉയർത്തി'
        }

    def extract_hand_landmarks(self, frame):
        """Extract hand landmarks from a frame"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                return np.array(landmarks)
        except Exception as e:
            st.error(f"Landmark extraction error: {e}")
        
        return None
    
    

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
    
    # Sidebar for instructions
    st.sidebar.header("Instructions")
    st.sidebar.info("""
    1. Click 'Start Camera'
    2. Show hand signs
    3. Translation will appear below video
    4. Select your preferred language from the dropdown
    """)

    # Camera control
    if st.checkbox('Start Camera', key='camera_checkbox'):
        if st.session_state.camera is None:
            st.session_state.camera = initialize_camera()
        
        # Create placeholders
        FRAME_WINDOW = st.empty()
        translation_container = st.empty()
        
        if st.session_state.camera and st.session_state.camera.isOpened():
            while True:
                ret, frame = st.session_state.camera.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Flip frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Extract landmarks
                landmarks = app.extract_hand_landmarks(frame)
                
                # Show translation when hand is detected
                if landmarks is not None:
                    translation = app.translations[selected_language]
                    translation_container.markdown(f"### Translation: {translation}")
                    
                    cv2.putText(frame, 
                        "Hand Detected", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2
                    )
                else:
                    translation_container.empty()
                
                # Display the frame
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Break if checkbox is unchecked
                if not st.session_state.camera_checkbox:
                    break
        else:
            st.error("Cannot open camera. Check connection and permissions.")
    else:
        # Release camera when checkbox is unchecked
        release_camera()

    # Back button (place at the bottom)
    if st.button("Back to Home"):
        release_camera()
        st.switch_page("Home.py")

if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure camera is released when the app is closed
        release_camera()
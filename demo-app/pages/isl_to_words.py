import streamlit as st
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import threading
import queue
import time

# Language translations dictionary
LANGUAGE_TRANSLATIONS = {
    'hello': {
        'English': 'Hello',
        'Hindi': 'नमस्ते',
        'Tamil': 'வணக்கம்',
        'Telugu': 'నమస్కారం',
        'Kannada': 'ನಮಸ್ಕಾರ',
        'Punjabi': 'ਸਤ ਸ੍ਰੀ ਅਕਾਲ',
        'Gujarati': 'નમસ્તે',
        'Marathi': 'नमस्कार',
        'Malayalam': 'ഹലോ'
    },
    'please': {
        'English': 'Please',
        'Hindi': 'कृपया',
        'Tamil': 'தயவு செய்து',
        'Telugu': 'దయచేసి',
        'Kannada': 'ದಯವಿಟ್ಟು',
        'Punjabi': 'ਕਿਰਪਾ ਕਰਕੇ',
        'Gujarati': 'કૃપા કરીને',
        'Marathi': 'कृपया',
        'Malayalam': 'ദയവായി'
    },
    'thankyou': {
        'English': 'Thank You',
        'Hindi': 'धन्यवाद',
        'Tamil': 'நன்றி',
        'Telugu': 'న్యాన్యాంక్యూ',
        'Kannada': 'ಧನ್ಯವಾದಗಳು',
        'Punjabi': 'ਧੰਨਵਾਦ',
        'Gujarati': 'આભાર',
        'Marathi': 'धन्यवाद',
        'Malayalam': 'നന്ദി'
    },
    'iloveyou': {
        'English': 'I Love You',
        'Hindi': 'मैं तुमसे प्यार करता हूँ',
        'Tamil': 'நான் உன்னை நேசிக்கிறேன்',
        'Telugu': 'నేను నిన్ను ప్రేమిస్తున్నాను',
        'Kannada': 'ನಾನು ನಿನ್ನನ್ನು ಪ್ರೀತಿಸುತ್ತೇನೆ',
        'Punjabi': 'ਮੈਂ ਤੈਨੂੰ ਪਿਆਰ ਕਰਦਾ ਹਾਂ',
        'Gujarati': 'હું તને પ્રેમ કરુ છુ',
        'Marathi': 'मी तुझ्यावर प्रेम करतो',
        'Malayalam': 'ഞാൻ നിന്നെ സ്നേഹിക്കുന്നു'
    },
    'yes': {
        'English': 'Yes',
        'Hindi': 'हाँ',
        'Tamil': 'ஆம்',
        'Telugu': 'అవును',
        'Kannada': 'ಹೌದು',
        'Punjabi': 'ਹਾਂ',
        'Gujarati': 'હા',
        'Marathi': 'होय',
        'Malayalam': 'അതെ'
    },
    'no': {
        'English': 'No',
        'Hindi': 'नहीं',
        'Tamil': 'இல்லை',
        'Telugu': 'కాదు',
        'Kannada': 'ಇಲ್ಲ',
        'Punjabi': 'ਨਹੀਂ',
        'Gujarati': 'ના',
        'Marathi': 'नाही',
        'Malayalam': 'വേണ്ട'
    }
}

class SignLanguageRecognizer:
    def __init__(self, api_url, api_key, model_id):
        self.client = InferenceHTTPClient(
            api_url=api_url,
            api_key=api_key
        )
        self.model_id = model_id
        self.prediction_queue = queue.Queue()
        self.prediction_thread = None
        self.stop_thread = threading.Event()

    def predict_frame(self, frame):
        # Save the current frame as a temporary file
        cv2.imwrite('current_frame.jpg', frame)

        try:
            # Perform inference
            result = self.client.infer('current_frame.jpg', model_id=self.model_id)
            return result
        except Exception as e:
            return {"error": str(e)}

    def prediction_worker(self, frame):
        result = self.predict_frame(frame)
        self.prediction_queue.put(result)

    def start_prediction(self, frame):
        # If a prediction thread is already running, wait for it to finish
        if self.prediction_thread and self.prediction_thread.is_alive():
            return None

        # Start a new prediction thread
        self.prediction_thread = threading.Thread(
            target=self.prediction_worker, 
            args=(frame,)
        )
        self.prediction_thread.start()
        return None

    def get_prediction(self):
        try:
            # Non-blocking check for prediction result
            return self.prediction_queue.get_nowait()
        except queue.Empty:
            return None

def main():
    st.title("Sign Language Recognition App")

    # Sidebar for API configuration
    st.sidebar.header("Configuration")
    api_url = st.sidebar.text_input("API URL", "https://detect.roboflow.com")
    api_key = st.sidebar.text_input("API Key", type="password")
    model_id = st.sidebar.text_input("Model ID", "sign-language-recognition-ryzbq/2")

    # Language selection
    selected_languages = st.multiselect(
        "Select Languages for Translation", 
        ['English', 'Hindi', 'Tamil', 'Telugu', 'Kannada', 'Punjabi', 'Gujarati', 'Marathi', 'Malayalam'],
        default=['English', 'Hindi']
    )

    # Prediction results display
    result_container = st.empty()
    translation_container = st.empty()

    # Camera input section
    run = st.checkbox('Start Camera')

    if run:
        # Open camera
        camera = cv2.VideoCapture(0)
        
        # Create placeholders for frame 
        frame_window = st.image([])
        predict_button = st.button('Predict Sign Language', key='predict_button')

        # Initialize recognizer
        recognizer = None
        if api_key:
            try:
                recognizer = SignLanguageRecognizer(api_url, api_key, model_id)
            except Exception as e:
                st.error(f"Failed to initialize recognizer: {e}")
                run = False

        # Frame rate control
        prev_frame_time = 0
        new_frame_time = 0

        # Continuous frame capture and display
        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture frame")
                break

            # Calculate FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            
            # Convert frame to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame
            frame_window.image(frame_rgb)

            # Prediction button logic
            if predict_button and recognizer:
                # Start prediction in a separate thread
                recognizer.start_prediction(frame)

            # Check for prediction results
            if recognizer:
                result = recognizer.get_prediction()
                if result:
                    if 'error' in result:
                        result_container.error(f"Prediction error: {result['error']}")
                    elif result and 'predictions' in result:
                        predictions = result['predictions']
                        if predictions:
                            # Display top prediction
                            top_prediction = predictions[0]
                            predicted_class = top_prediction.get('class', 'Unknown').lower()
                            
                            # Display top prediction with large bold text
                            result_container.markdown(f"<h1 style='text-align: center; font-weight: bold; font-size: 50px;'>{predicted_class}</h1>", unsafe_allow_html=True)
                            
                            # Display translations if the predicted class is in our dictionary
                            if predicted_class in LANGUAGE_TRANSLATIONS:
                                # Create translation display
                                translations = []
                                for lang in selected_languages:
                                    translation = LANGUAGE_TRANSLATIONS[predicted_class].get(lang, 'Not Available')
                                    translations.append(f"**{lang}**: {translation}")
                                
                                # Join translations with line breaks
                                translation_text = "<br>".join(translations)
                                translation_container.markdown(f"<div style='text-align: center; font-size: 20px;'>{translation_text}</div>", unsafe_allow_html=True)
                            else:
                                # Clear translations if no translation available
                                translation_container.empty()
                        else:
                            result_container.warning("No signs detected")

            # Optional: display FPS
            st.sidebar.text(f"FPS: {fps:.2f}")

        # Cleanup
        camera.release()
    else:
        st.info("Check 'Start Camera' to begin")

if __name__ == "__main__":
    main()
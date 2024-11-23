Web Interface to demo the pre-trained machine learning models with feature prototypes. Built using Streamlit python library.

## How to Run
1. Install dependencies
```python
pip install streamlit opencv-python mediapipe tensorflow gTTS
```
2. Start the app (make sure you are in /demo-app directory)
```python
streamlit run Home.py
```
3. Go to http://localhost:8501

### Folder Structure
```
demo-app/
│
├── Home.py
│
└── pages/
    ├── isl_to_text.py
    └── text_to_isl.py
```
import streamlit as st

st.set_page_config(
    page_title="Sanketuvach Demo"
)

def home_page():
    st.title("🙏 Sanketuvach Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("ISL to Text")
        if st.button("Convert Sign Language to Text"):
            st.switch_page("pages/isl_to_text.py")
    
    with col2:
        st.header("Text to ISL")
        if st.button("Convert Text to Sign Language"):
            st.switch_page("pages/text_to_isl.py")

    with col2:
        st.header("Mediapipe test")
        if st.button("Count fingers"):
            st.switch_page("pages/test_app.py")

def main():
    home_page()

if __name__ == "__main__":
    main()
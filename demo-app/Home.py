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

    with col1:
        st.header("ISL to Words")
        if st.button("ISL to Words."):
            st.switch_page("pages/isl_to_words.py")
    
    with col2:
        st.header("Text to ISL")
        if st.button("Convert Text to Sign Language"):
            st.switch_page("pages/text_to_isl.py")

def main():
    home_page()

if __name__ == "__main__":
    main()
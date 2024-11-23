import streamlit as st

def main():
    st.title("Text to ISL Converter")
    
    # Back button
    if st.button("Back to Home"):
        st.switch_page("Home.py")
    
    st.write("Coming Soon!")

if __name__ == "__main__":
    main()
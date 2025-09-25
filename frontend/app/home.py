import streamlit as st
from pathlib import Path
import requests
from train import train_app
from visualization import viz_data
import uuid


st.set_page_config(page_title="ML Playground 🚀", layout="wide")

# Initialize session state for page
if "page" not in st.session_state:
    st.session_state.page = "home"
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'uuid' not in st.session_state:
    st.session_state.uuid = str(uuid.uuid4())

# Sidebar Navigation
st.sidebar.title("Navigation")
if st.sidebar.button("🏠 Home"):
    st.session_state.page = "home"
if st.sidebar.button("🛠 Train Model"):
    st.session_state.page = "train"
if st.sidebar.button("📊 Visualize Data"):
    st.session_state.page = "visualize"
if st.sidebar.button("⬆️ Upload Data"):
    st.session_state.page = "upload_data"

# Page Routing
if st.session_state.page == "home":
    st.title("🚀 ML Playground")
    st.markdown("### Explore, visualize, and train ML models with your dataset")
    st.markdown(
        """
        Welcome to the **ML Playground**!  
        Use the sidebar to:
        - 🛠 Train ML models  
        - 📊 Visualize data  
        """
    )

elif st.session_state.page == "train":
   train_app()

elif st.session_state.page == "visualize":
   viz_data(st.session_state)

elif st.session_state.page == "upload_data":
    st.title("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        try:
            response = requests.post("http://ml-playground-backend-service:8000/upload_file", files = {"file": uploaded_file}, data = {"uuid": st.session_state.uuid})
            if response.status_code == 200:
                response_json = response.json()
                if 'message' in response_json:
                    st.info(response_json['message'])
                elif 'error' in response_json:
                    st.error(response_json['error'])
            else:
                st.error("Could not upload file")
        except Exception as e:
            st.error(f"Request failed: {e}")
    



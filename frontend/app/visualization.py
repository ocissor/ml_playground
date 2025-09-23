import streamlit as st
import requests
import json
import base64
from io import BytesIO  
from PIL import Image
import pickle

def viz_data(app_state):
    
    st.set_page_config(page_title="ML Playground 🚀", layout="wide")
    st.title("🚀 ML Playground")
    st.markdown("### Explore and visualize your dataset")

    # ------------------- Data Exploration ------------------- #
    if app_state.uploaded_file:
        st.subheader("🔍 Data Exploration")
        
        col1, col2 = st.columns([1, 2])

        with col1:
            if st.button("Get Data Stats"):
                try:
                    app_state.uploaded_file.seek(0)
                    response = requests.post("http://127.0.0.1:8000/data_stats", files={"file": app_state.uploaded_file})
                    if response.status_code == 200:
                        response_json = response.json()
                        if 'data_stats' in response_json:
                            st.table(json.loads(response_json['data_stats']))
                        else:
                            st.info("No data stats available")
                    else:
                        st.error("Could not retrieve data stats")
                except Exception as e:
                    st.error(f"Request failed: {e}")

        with col2:
            if st.button("Visualize Data"):
                try:
                    app_state.uploaded_file.seek(0)
                    response = requests.post("http://127.0.0.1:8000/visualize_data", files={"file": app_state.uploaded_file})
                    if response.status_code == 200:
                        images = response.json()
                        if images:
                            num_cols = 2
                            cols = st.columns(num_cols)
                            for idx, (title, img_bytes) in enumerate(images.items()):
                                img = Image.open(BytesIO(base64.b64decode(img_bytes)))
                                with cols[idx % num_cols]:
                                    st.image(img, caption=title, use_container_width=True)
                        else:
                            st.info("No visualizations available for the provided data.")
                    else:
                        st.error("Could not retrieve visualizations")
                except Exception as e:
                    st.error(f"Request failed: {e}")

    else:
        st.info("Please upload a dataset in the 'Upload Data' section from the sidebar to Vizualize data.")

    user_input = st.chat_input("Ask any query related to data", key="data_query")

    if user_input:
        st.chat_message("user").markdown(user_input)
        if app_state.uploaded_file:
            with st.spinner("Generating response..."):
                try:
                    app_state.uploaded_file.seek(0)
                    response = requests.post("http://127.0.0.1:8000/chat_with_data", files = {"file":app_state.uploaded_file}, data = {"user_input":user_input, 'uuid':app_state.uuid})
                    if response.status_code == 200:
                        output = pickle.loads(response.content)
                        for message in reversed(output['messages'][-2:]):
                            if message.type == 'ai':
                                st.text(message.content.replace("\\n", "\n"))
                            elif message.type == 'tool':
                                if message.content.startswith("data:image"):
                                    img_bytes = base64.b64decode(message.content.split(',')[1])
                                    img = Image.open(BytesIO(img_bytes))
                                    st.image(img, use_container_width=True)
                                else:
                                    st.text(message.content.replace("\\n", "\n"))
                except Exception as e:
                    st.error(f"Request failed: {e}")
        else:
            st.info("Please upload a dataset in the 'Upload Data' section from the sidebar to chat with data.")

 

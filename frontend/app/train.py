import streamlit as st
import requests
import json
from config import model_names, model_hyperparams, model_hyperparams_values
from helper import generate_hyperparameters_widget


def train_app():

    st.set_page_config(page_title="ML Playground 🚀", layout="wide")
    st.title("🚀 ML Playground")
    st.markdown("### Explore, visualize, and train ML models with your dataset")

    # ------------------- Data Exploration ------------------- #

    # ------------------- Model Training ------------------- #
    if st.session_state.uploaded_file:
        st.subheader("🛠 Model Training")

        # Target column & model selection
        target_column = st.text_input("Enter the target column name (label):")
        model_choice = st.selectbox("Choose Model", model_names, key="model_choice")
        problem_type = st.selectbox("Problem Type", ["classification", "regression"])

        if model_choice:
            with st.form("model_params_form"):
                st.markdown("#### ⚙ Model Hyperparameters")
                
                params = model_hyperparams.get(model_choice, {})
                l1 = set([key.lower() for key in list(model_hyperparams_values.get(model_choice, {}).keys())])
                l2 = set([key.lower() for key in list(params.keys())])
                common_keys = l2.intersection(l1)
                filtered_params = {key: params[key] for key in params if key.lower() in common_keys}
                params = generate_hyperparameters_widget(filtered_params, model_choice)
                
                submitted = st.form_submit_button("Save Hyperparameters")
                if submitted:
                    st.success("Hyperparameters saved!")

            # Train button
            if st.button("Train Model"):
                if target_column:
                    data = {
                        "target_column": target_column,
                        "model_name": model_choice,
                        "problem_type": problem_type,
                        "hyperparameter": json.dumps(params)
                    }
                    try:
                        response = requests.post("http://127.0.0.1:8000/train", files={"file": st.session_state.uploaded_file}, data=data)
                        if response.status_code == 200:
                            result = response.json()
                            if "error" in result:
                                st.error(result["error"])
                            else:
                                st.success("Model trained successfully!")
                                st.success(f"accuracy of the model is : {result['model_outputs'][model_choice]['accuracy']}")
                        else:
                            st.error("Unable to train model. Check target column or dataset.")
                    except Exception as e:
                        st.error(f"Request failed: {e}")
                else:
                    st.warning("Please enter the target column name.")

    else:
        st.info("Please upload a dataset in the 'Upload Data' section from the sidebar to train models.")

    # ------------------- Footer ------------------- #
    st.markdown("---")
    

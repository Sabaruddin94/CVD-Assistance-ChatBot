import streamlit as st
import requests
import json
import logging
import numpy as np
from typing import Optional
from PIL import Image
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os


# Constant
BASE_API_URL = "https://0132-175-139-159-165.ngrok-free.app"
FLOW_ID = "eaf5c685-5e3b-450d-80ac-40096ce0681c"
ENDPOINT = "" # You can set a specific endpoint name in the flow settings
TWEAKS = {
  "ChatInput-feB0F": {},
  "ChatOutput-vikhM": {},
  "GoogleGenerativeAIModel-imboU": {},
  "Prompt-upBoA": {},
  "Memory-1YMJe": {},
  "TextInput-KlD03": {},
  "TextInput-ZmmJM": {}
}

# Function to run the flow
# Initialize logging
logging.basicConfig(level=logging.INFO)


# Function to run the flow
def run_flow(message: str,
             endpoint: str = FLOW_ID,
             output_type: str = "chat",
             input_type: str = "chat",
             tweaks: Optional[dict] = None,
             api_key: Optional[str] = None) -> dict:
    """
    Run a flow with a given message and optional tweaks.

    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param tweaks: Optional tweaks to customize the flow
    :return: The JSON response from the flow
    """
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }

    if tweaks:
        payload["tweaks"] = tweaks

    headers = {"x-api-key": api_key} if api_key else None
    response = requests.post(api_url, json=payload, headers=headers)

    # Log the response for debugging
    logging.info(f"Response Status Code: {response.status_code}")
    logging.info(f"Response Text: {response.text}")

    try:
        return response.json()
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON from the server response.")
        return {}


# Function to extract the assistant's message from the response
def extract_message(response: dict) -> str:
    try:
        # Extract the response message
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        logging.error("No valid message found in response.")
        return "No valid message found in response."
    
@st.cache_resource  # Cache the model to avoid reloading on each run
def load_model():
    model = os.path.join(os.getcwd(), 'model', 'model.keras') 
    if os.path.exists(model):
        model = tf.keras.models.load_model(model)
    return model




model = load_model()

# Function to run the flow
def main():
    # Initialize session state for patients DataFrame
    if 'patients' not in st.session_state:
        # Initialize an empty DataFrame with columns
        st.session_state.patients = pd.DataFrame(columns=["Patient ID", "Name", "Condition", "Treatment Plan"])
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize session state for patients DataFrame
    if 'patients' not in st.session_state:
        # Initialize an empty DataFrame with columns
        st.session_state.patients = pd.DataFrame(columns=["Patient ID", "Name", "Condition", "Treatment Plan"])

    
    # Display the image at the top of your app using st.image() from a local file
    st.image("static/blockage1.jpg", use_container_width=True)
    
    st.markdown('<h1 class="title">Welcome to the CVD Info Chatbot! 🤖</h1>', unsafe_allow_html=True)
    
    # Create a sidebar
    with st.sidebar:
        youtube_url = "https://www.youtube.com/watch?v=njT428_JYzI"
        st.video(youtube_url)
        st.image("static/blockage2.jpg", use_container_width=True)
        st.markdown("""<h3>About Artery Stenosis Classifier</h3><p>This app uses AI to classify MRI images of arteries as healthy or stenosed.</p>""", unsafe_allow_html=True)
        st.markdown("### Features:")
        st.markdown("""- AI-powered artery stenosis classification\n- Instant prediction for uploaded MRI images\n- Available 24/7""")

    # Sidebar for file uploader
    with st.sidebar:
        st.header("Upload your image here!")
        st.markdown("For the best and most accurate results, please make sure the image is clear zoom at the specific bifurcated artery and contains only one type of item at a time. Thanks! 😊")
        image = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

        if image is not None:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            image = Image.open(image)
            image = image.resize((300, 300), Image.Resampling.LANCZOS) # or other resampling methods
            image = np.array(image)
            image = np.expand_dims(image, axis=0)

            # Predict the class probabilities (not just the index)
            predictions = model.predict(image)

            # Get the class index with the highest probability
            predict = np.argmax(predictions)

            # Get the actual probabilities for each class
            healthy_prob = predictions[0][0]  # Probability of "Healthy Artery"
            unhealthy_prob = predictions[0][1]  # Probability of "Unhealthy Artery"
            
            st.write(f"Healthy Probability: {healthy_prob}")
            st.write(f"Unhealthy Probability: {unhealthy_prob}")
            
            # Define class names
            class_names = ['Healthy Artery with no Stenosis progression. Prevention', 'Unhealthy. Stenosis Progression. Treatment and medical cost in Malaysia vs other country']
            predicted_class = class_names[predict]
            
            # ✅ Check if the class info was already added to avoid duplicates
            if not any(msg["content"] == f"{predicted_class}?" for msg in st.session_state.messages):
                # Save predicted class as a user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"{predicted_class}?",
                    "avatar": "💬"
                })

                # Get assistant response and save it
                assistant_response = extract_message(run_flow(f"{predicted_class}?", tweaks=TWEAKS))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response,
                    "avatar": "🤖"
                })
    
    # Function to add right side panel with User Rating, Feedback, and Social Media Links
    with st.sidebar:
        # Right sidebar panel with user statistics
        st.sidebar.header("🔍 User Statistics")
        today = datetime.date.today()
        st.sidebar.write(f"**Users Today**: 15")
        st.sidebar.write(f"**Total Users**: 500")
        st.sidebar.write(f"**Active Sessions**: 3")
        st.sidebar.write(f"**Date**: {today}")
    
        # User Rating / Feedback Widget
        st.sidebar.header("🌟 User Rating")
        rating = st.sidebar.slider('Rate our app', min_value=1, max_value=5, value=3)
        st.sidebar.write(f"Your Rating: {rating} Stars")

        # Feedback section
        st.sidebar.header("📝 Feedback")
        feedback = st.sidebar.text_area("Leave your feedback or suggestion", "")
        if st.sidebar.button('Submit Feedback'):
            if feedback:
                st.sidebar.success("Thank you for your feedback!")
            else:
                st.sidebar.warning("Please write some feedback before submitting.")
    

    # Display previous messages with avatars
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.write(message["content"])

    # Input box for user message
    if query := st.chat_input("Ask me anything..."):
        # Add user message to session state
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
                "avatar": "💬",  # Emoji for user
            }
        )
        with st.chat_message("user", avatar="💬"):  # Display user message
            st.write(query)

         # Call the Langflow API and get the assistant's response
        with st.chat_message("assistant", avatar="🤖"):  # Emoji for assistant
            message_placeholder = st.empty()  # Placeholder for assistant response
            with st.spinner("Thinking..."):
                # Fetch response from Langflow with updated TWEAKS and using `query`
                assistant_response = extract_message(run_flow(query, tweaks=TWEAKS))
                message_placeholder.write(assistant_response)

        # Add assistant response to session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
                "avatar": "🤖",  # Emoji for assistant
            }
        )
st.title("Understanding the CVD Chatbot: Early Detection of Stenosed Bifurcated Artery and Its Treatment  🤖")

# Add patient data entry form
st.header("📝 Patient Data Entry Management")

# Ensure that 'patients' is initialized in session_state to avoid errors when the form is submitted
if 'patients' not in st.session_state:
    st.session_state.patients = pd.DataFrame(columns=["Patient ID", "Name", "Age", "Condition", "Treatment Plan", "Cholesterol Level","Blood Pressure","Diabetes","Smoking"])

with st.form(key='patient_form'):
    # Input fields for patient information
    patient_id = st.text_input("Patient ID")
    patient_name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=2)  # Added Age field
    condition = st.selectbox("Condition", ["Healthy", "Stenosed"])
    treatment_plan = st.selectbox("Treatment Plan", ["Medications", "Diet","Angioplasty","Bypass Surgery","Antiplatelet Agents"])
    Cholesterol = st.number_input('Cholestrol Level (mg/dL)', min_value=0, max_value=300, value=10)
    Blood_Pressure = st.number_input('Reading (mmHg)', min_value=0, max_value=300, value=20)
    Diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    Smoking = st.selectbox("Smoking", ["Regular", "Social Smoker","No"])

    
    # Submit button
    submit_button = st.form_submit_button(label="Save Patient Data")

    # Check if the form is submitted
    if submit_button:
        # Basic validation to ensure the required fields are filled
        if not patient_id or not patient_name:
            st.warning("Patient ID and Patient Name are required!")
        else:
            # Create a new patient record
            new_patient = {
                "Patient ID": patient_id,
                "Name": patient_name,
                "Age": age,
                "Condition": condition,
                "Treatment Plan": treatment_plan,
                "Cholesterol": Cholesterol,
                "Blood_Pressure": Blood_Pressure,
                "Diabetes": Diabetes,
                "Smoking":Smoking

            }
            
            # Convert new patient info to a DataFrame and append to session_state.patients
            new_patient_df = pd.DataFrame([new_patient])
            st.session_state.patients = pd.concat([st.session_state.patients, new_patient_df], ignore_index=True)
            st.success("Patient Data Saved!")

# Display patient data as a table
st.subheader("Patient Data")
if st.session_state.patients.empty:
    st.write("No patient data available.")
else:
    st.dataframe(st.session_state.patients)

# Visualizations: Age vs. Condition, Age vs. Treatment Plan, and Condition vs. Treatment Plan

# Check if there is enough data to plot
if not st.session_state.patients.empty:
    # Plot Age vs. Condition
    st.subheader("Age vs. Condition")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="Condition", y="Age", data=st.session_state.patients, ax=ax)
    ax.set_title("Age vs. Condition")
    st.pyplot(fig)

    # Plot Age vs. Treatment Plan
    st.subheader("Age vs. Treatment Plan")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="Treatment Plan", y="Age", data=st.session_state.patients, ax=ax)
    ax.set_title("Age vs. Treatment Plan")
    st.pyplot(fig)

    # Plot Condition vs. Treatment Plan
    st.subheader("Condition vs. Treatment Plan")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Condition", hue="Treatment Plan", data=st.session_state.patients, ax=ax)
    ax.set_title("Condition vs. Treatment Plan")
    st.pyplot(fig)

# Footer Section (Optional)
st.markdown('<div class="footer">© 2025 Artery Stenosis Classifier. All rights reserved.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Set page configuration
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Chest X-Ray Pneumonia Detection")
st.write("""
Upload a chest X-ray image to detect the possible presence of pneumonia.
""")

# --- MODEL LOADING ---
# Uncomment and update the path to your actual trained model
# @st.cache_resource
# def load_pneumonia_model():
#     model = tf.keras.models.load_model('model.h5')
#     return model

# model = load_pneumonia_model()

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose a Chest X-Ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Chest X-Ray', use_container_width=True)
    
    st.write("")
    
    # Prediction button
    if st.button("Run Prediction", type="primary"):
        with st.spinner('Analyzing image...'):
            try:
                # --- PREPROCESSING ---
                # Adjust the target size to match what your model expects (e.g., 224x224)
                target_size = (224, 224)
                img = image.resize(target_size)
                
                img_array = np.array(img)
                img_array = img_array / 255.0  # Normalize pixel values
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                
                # --- PREDICTION ---
                # Uncomment the following lines when using your actual model
                # prediction = model.predict(img_array)
                # probability = prediction[0][0] 
                
                # --- MOCK PREDICTION ---
                # This is a placeholder for demonstration. Remove this block when you plug in your model.
                import random
                import time
                time.sleep(1.5) # Simulate processing time
                probability = random.uniform(0, 1)
                # -----------------------
                
                # Determine class based on probability (threshold typically 0.5)
                # Note: Adjust logic based on how your model was trained (e.g. 0=Normal, 1=Pneumonia)
                st.write("### Prediction Results")
                
                pneumonia_pct = probability * 100
                normal_pct = (1 - probability) * 100
                
                # Display percentages explicitly
                st.write(f"- **Pneumonia Probability:** {pneumonia_pct:.2f}%")
                st.write(f"- **Normal Probability:** {normal_pct:.2f}%")
                st.write("")
                
                if probability > 0.5:
                    st.error(f"**Final Verdict: Pneumonia Detected**")
                    st.warning("⚠️ **Note:** This is an AI prediction. Please consult a healthcare professional for a proper diagnosis.")
                else:
                    st.success(f"**Final Verdict: Normal**")
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

# Footer disclaimer
st.markdown("---")
st.caption("*Disclaimer: This application is a prototype for educational and demonstration purposes only. It is not intended for use in the diagnosis of disease or other conditions, or in the cure, mitigation, treatment, or prevention of disease.*")

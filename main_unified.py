import streamlit as st
import tensorflow as tf
import numpy as np
from hyperspectral_utils import HyperspectralPreprocessor
import os
from disease_info_display import display_disease_info
import pandas as pd

# Initialize the hyperspectral preprocessor
preprocessor = HyperspectralPreprocessor(target_size=(128, 128), n_bands=100)

# --- Sidebar Enhancements ---
# Add a logo or project image at the top
# st.sidebar.image("home_page.jpeg", use_container_width=True)

# Add navigation links or quick info
# st.sidebar.title("🌱 Plant Disease Recognition")



def predict_rgb_image(image_path):
    """Prediction for regular RGB images"""
    try:
        # Load the model
        model = tf.keras.models.load_model('trained_model.h5')
        
        # Load and preprocess the image
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch
        
        # Make prediction
        prediction = model.predict(input_arr)
        return np.argmax(prediction)
    except Exception as e:
        st.error(f"Error in RGB image prediction: {str(e)}")
        return None

def predict_hyperspectral_image(image_path):
    """Prediction for hyperspectral images"""
    try:
        model = tf.keras.models.load_model('trained_model_hyperspectral.h5')
        input_arr = preprocessor.preprocess_for_model(image_path)
        prediction = model.predict(input_arr)
        return np.argmax(prediction)
    except Exception as e:
        st.error(f"Error processing hyperspectral image: {str(e)}")
        return None

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])


# # Add a theme switcher (light/dark mode)
# theme = st.sidebar.radio(
#     "Theme Mode",
#     ("Light", "Dark"),
#     index=0,
#     help="Switch between light and dark mode."
# )

# Apply theme (Streamlit doesn't support dynamic theme switching natively, but we can give a visual cue)
# if theme == "Dark":
#     st.markdown(
#         """
#         <style>
#         body, .stApp { background-color: #222 !important; color: #eee !important; }
#         .css-1d391kg { background-color: #222 !important; }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
# else:
#     st.markdown(
#         """
#         <style>
#         body, .stApp { background-color: #fff !important; color: #222 !important; }
#         .css-1d391kg { background-color: #fff !important; }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to our Advanced Plant Disease Recognition System! 🌿🔍
    
    Our system supports both traditional RGB images and advanced hyperspectral imaging for plant disease detection.
    """)

    # --- Sample Images Grid ---
    st.subheader("Sample Plant Images")
    sample_images = [
        ("Apple Scab", "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"),
        ("Tomato Early Blight", "https://images.unsplash.com/photo-1506744038136-46273834b3fb"),
        ("Potato Late Blight", "https://images.unsplash.com/photo-1465101046530-73398c7f28ca"),
        ("Grape Black Rot", "https://images.unsplash.com/photo-1519125323398-675f0ddb6308"),
    ]
    cols = st.columns(4)
    for i, (label, url) in enumerate(sample_images):
        with cols[i]:
            st.image(url, caption=label, use_container_width=True)

    # --- Get Started Button ---
    st.markdown("""
    <div style='text-align: center; margin-top: 2em;'>
        <a href='#disease-recognition' style='text-decoration: none;'>
            <button style='background-color: #4CAF50; color: white; padding: 1em 2em; border: none; border-radius: 8px; font-size: 1.2em; cursor: pointer;'>
                🚀 Get Started
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    ### Features
    1. **Dual Image Processing:**
       - Traditional RGB image analysis
       - Advanced hyperspectral image analysis
    2. **Benefits:**
       - RGB Images: Quick and accessible disease detection
       - Hyperspectral Images: Early detection and higher accuracy
    ### How to Use
    1. Go to the **Disease Recognition** page
    2. Choose your image type (RGB or Hyperspectral)
    3. Upload your image
    4. Get instant analysis results
    """)

    # --- Contact Us / Feedback Section ---
    st.markdown("---")
    st.subheader("Contact Us / Feedback")
    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message or Feedback")
        submitted = st.form_submit_button("Send")
        if submitted:
            st.success("Thank you for your feedback! We appreciate your input.")

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    ### Our Technology
    
    This system combines two powerful imaging technologies:
    
    #### 1. RGB Imaging
    - Traditional digital photography
    - Accessible and widely available
    - Good for visible disease symptoms
    
    #### 2. Hyperspectral Imaging
    - Captures hundreds of spectral bands
    - Early disease detection
    - Higher accuracy for complex cases
    - Non-destructive analysis
    
    ### How It Works
    The system uses deep learning models trained on both RGB and hyperspectral datasets to provide accurate disease detection.
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    
    # Initialize session state for history and clear
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'clear' not in st.session_state:
        st.session_state['clear'] = False

    # Image type selection
    image_type = st.radio("Select Image Type", ["RGB Image", "Hyperspectral Image"])
    
    # File uploader
    test_image = None
    if image_type == "RGB Image":
        test_image = st.file_uploader("Choose an RGB Image:", type=['jpg', 'jpeg', 'png'], key='rgb_upload')
    else:
        test_image = st.file_uploader("Choose a Hyperspectral Image:", type=['hdr', 'raw', 'hyp'], key='hyp_upload')
    
    # Show preview of uploaded image
    if test_image is not None:
        st.subheader("Preview of Uploaded Image")
        st.image(test_image, use_container_width=True)

    # Clear button
    if st.button("Clear"):
        st.session_state['clear'] = True
        st.session_state['history'] = []
        st.experimental_rerun()

    # Predict button
    if test_image is not None and st.button("Predict"): 
        with st.spinner("Analyzing image..."):
            # Define class names
            class_names = ['Apple___Apple_scab',
                         'Apple___Black_rot',
                         'Apple___Cedar_apple_rust',
                         'Apple___healthy',
                         'Blueberry___healthy',
                         'Cherry_(including_sour)___Powdery_mildew',
                         'Cherry_(including_sour)___healthy',
                         'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                         'Corn_(maize)___Common_rust_',
                         'Corn_(maize)___Northern_Leaf_Blight',
                         'Corn_(maize)___healthy',
                         'Grape___Black_rot',
                         'Grape___Esca_(Black_Measles)',
                         'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                         'Grape___healthy',
                         'Orange___Haunglongbing_(Citrus_greening)',
                         'Peach___Bacterial_spot',
                         'Peach___healthy',
                         'Pepper,_bell___Bacterial_spot',
                         'Pepper,_bell___healthy',
                         'Potato___Early_blight',
                         'Potato___Late_blight',
                         'Potato___healthy',
                         'Raspberry___healthy',
                         'Soybean___healthy',
                         'Squash___Powdery_mildew',
                         'Strawberry___Leaf_scorch',
                         'Strawberry___healthy',
                         'Tomato___Bacterial_spot',
                         'Tomato___Early_blight',
                         'Tomato___Late_blight',
                         'Tomato___Leaf_Mold',
                         'Tomato___Septoria_leaf_spot',
                         'Tomato___Spider_mites Two-spotted_spider_mite',
                         'Tomato___Target_Spot',
                         'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                         'Tomato___Tomato_mosaic_virus',
                         'Tomato___healthy']
            
            # Save uploaded file temporarily
            temp_path = f"temp_image.{test_image.name.split('.')[-1]}"
            with open(temp_path, "wb") as f:
                f.write(test_image.getbuffer())
            
            try:
                # Process based on image type
                if image_type == "RGB Image":
                    model = tf.keras.models.load_model('trained_model.h5')
                    image = tf.keras.preprocessing.image.load_img(temp_path, target_size=(128, 128))
                    input_arr = tf.keras.preprocessing.image.img_to_array(image)
                    input_arr = np.array([input_arr])
                    prediction = model.predict(input_arr)[0]
                else:
                    model = tf.keras.models.load_model('trained_model_hyperspectral.h5')
                    input_arr = preprocessor.preprocess_for_model(temp_path)
                    prediction = model.predict(input_arr)[0]
                result_index = int(np.argmax(prediction))
                predicted_disease = class_names[result_index]
                confidence = float(np.max(prediction))
                
                # Show prediction and confidence
                st.success(f"Prediction: {predicted_disease}")
                st.write(f"**Confidence:** {confidence*100:.2f}%")
                
                # Show prediction probabilities as a bar chart
                prob_df = pd.DataFrame({
                    'Disease': class_names,
                    'Probability': prediction
                }).sort_values('Probability', ascending=False).head(5)
                st.bar_chart(prob_df.set_index('Disease'))
                
                # Display disease information if it's not a healthy plant
                if "healthy" not in predicted_disease.lower():
                    st.markdown("---")
                    st.header("📚 Disease Information")
                    display_disease_info(predicted_disease)
                
                # Add to history
                st.session_state['history'].append({
                    'disease': predicted_disease,
                    'confidence': confidence,
                    'image_name': test_image.name
                })
                
                # Download PDF report (simple text-based for now)
                import io
                from fpdf import FPDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Plant Disease Recognition Report", ln=True, align='C')
                pdf.ln(10)
                pdf.cell(200, 10, txt=f"Image: {test_image.name}", ln=True)
                pdf.cell(200, 10, txt=f"Prediction: {predicted_disease}", ln=True)
                pdf.cell(200, 10, txt=f"Confidence: {confidence*100:.2f}%", ln=True)
                pdf.ln(10)
                pdf.cell(200, 10, txt="See app for detailed disease info.", ln=True)
                pdf_bytes = pdf.output(dest='S').encode('latin1')
                st.download_button(
                    label="Download Report as PDF",
                    data=pdf_bytes,
                    file_name=f"{test_image.name}_report.pdf",
                    mime="application/pdf"
                )
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    # --- History Section ---
    if st.session_state['history']:
        st.markdown("---")
        st.subheader("🕑 Prediction History (this session)")
        for i, entry in enumerate(reversed(st.session_state['history'])):
            st.write(f"{i+1}. **{entry['disease']}** ({entry['confidence']*100:.2f}%) - {entry['image_name']}")

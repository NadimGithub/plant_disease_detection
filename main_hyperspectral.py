import streamlit as st
import tensorflow as tf
import numpy as np
from hyperspectral_utils import HyperspectralPreprocessor

# Initialize the hyperspectral preprocessor
# Adjust n_bands based on your hyperspectral data
preprocessor = HyperspectralPreprocessor(target_size=(128, 128), n_bands=100)

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model_hyperspectral.h5')
    
    # Preprocess hyperspectral image
    try:
        input_arr = preprocessor.preprocess_for_model(test_image)
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        return result_index
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Home Page
if(app_mode=="Home"):
    st.header("HYPERSPECTRAL PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Hyperspectral Plant Disease Recognition System! 🌿🔍
    
    Our advanced system uses hyperspectral imaging technology to detect plant diseases with higher accuracy. 
    Hyperspectral imaging captures information across the electromagnetic spectrum, allowing us to detect disease symptoms 
    before they become visible to the naked eye.

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload a hyperspectral image of a plant.
    2. **Analysis:** Our system processes the hyperspectral data using advanced deep learning algorithms.
    3. **Results:** View detailed analysis results including disease classification.

    ### Advantages of Hyperspectral Imaging
    - **Early Detection:** Identify diseases before visible symptoms appear
    - **Higher Accuracy:** Multiple spectral bands provide more detailed information
    - **Non-destructive:** Analysis without damaging the plant
    """)

#About Page
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
    #### About the Technology
    This system uses hyperspectral imaging technology, which captures hundreds of continuous spectral bands for each pixel in the image. 
    This provides significantly more information than traditional RGB images, allowing for:
    
    1. Early disease detection
    2. More accurate diagnosis
    3. Detection of stress factors
    4. Nutrient deficiency analysis
    
    #### Dataset
    The hyperspectral dataset contains plant images captured across multiple spectral bands, 
    providing detailed spectral signatures of healthy and diseased plants.
    """)
    
#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose a Hyperspectral Image:")
    
    if test_image is not None:
        if(st.button("Show Image")):
            # Display a preview of the hyperspectral image
            # Note: You'll need to implement proper visualization for hyperspectral data
            st.write("Displaying preview of hyperspectral image (showing RGB channels)")
            
        #Predict Button
        if(st.button("Predict")):
            with st.spinner("Analyzing hyperspectral data..."):
                result_index = model_prediction(test_image)
                if result_index is not None:
                    #Define Class Names
                    class_names = ['Apple___Apple_scab',
                                'Apple___Black_rot',
                                'Apple___Cedar_apple_rust',
                                'Apple___healthy',
                                # ... (keep your existing class names)
                                ]
                    st.success(f"Analysis Complete: {class_names[result_index]}")
                    
                    # Additional spectral analysis information
                    st.write("### Spectral Analysis")
                    st.write("Displaying key spectral features that contributed to the diagnosis...")
                    # Add visualization of important spectral bands here

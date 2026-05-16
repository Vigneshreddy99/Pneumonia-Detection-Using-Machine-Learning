"""
Pneumonia Detection Web Application
A Streamlit-based web interface for pneumonia detection from chest X-rays
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import yaml
import requests
from io import BytesIO
import cv2

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_resource
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Load model
@st.cache_resource
def load_pneumonia_model():
    try:
        model = load_model(config['deployment']['model_path'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_pneumonia_model()

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 3rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .normal {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .pneumonia {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize image
    img = image.resize((config['data']['img_height'], config['data']['img_width']))
    
    # Convert to array
    img_array = np.array(img)
    
    # Convert to RGB if grayscale
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Normalize
    img_array = img_array * config['augmentation']['rescale']
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_pneumonia(image):
    """Make prediction on the image"""
    if model is None:
        return None, None
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    # Get class name
    class_names = config['data']['classes']
    predicted_label = class_names[predicted_class]
    
    return predicted_label, confidence, predictions[0]

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Pneumonia Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Chest X-Ray Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.info("""
        This application uses deep learning (VGG16) to detect pneumonia from chest X-ray images.
        
        **Classes:**
        - Normal
        - Bacterial Pneumonia
        - Viral Pneumonia
        
        **How to use:**
        1. Upload an X-ray image or provide a URL
        2. Click 'Analyze X-Ray'
        3. View the prediction results
        """)
        
        st.header("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=config['deployment']['confidence_threshold'],
            step=0.05
        )
        
        st.header("ℹ️ Model Info")
        st.write(f"**Model:** {config['model']['name']}")
        st.write(f"**Base Architecture:** {config['model']['base_model']}")
        st.write(f"**Input Size:** {config['data']['img_height']}x{config['data']['img_width']}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload X-Ray Image")
        
        # Input method selection
        input_method = st.radio("Choose input method:", ["Upload File", "Image URL"])
        
        image = None
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a chest X-ray image...",
                type=["jpg", "jpeg", "png"],
                help="Upload a chest X-ray image in JPG, JPEG, or PNG format"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                
        else:  # Image URL
            image_url = st.text_input(
                "Enter image URL:",
                placeholder="https://example.com/xray.jpg"
            )
            
            if image_url:
                try:
                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                except Exception as e:
                    st.error(f"Error loading image from URL: {e}")
        
        if image:
            st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)
            
            # Analyze button
            if st.button("🔍 Analyze X-Ray", type="primary"):
                with st.spinner("Analyzing image..."):
                    prediction, confidence, all_predictions = predict_pneumonia(image)
                    
                    if prediction:
                        # Store results in session state
                        st.session_state['prediction'] = prediction
                        st.session_state['confidence'] = confidence
                        st.session_state['all_predictions'] = all_predictions
                        st.session_state['threshold'] = confidence_threshold
    
    with col2:
        st.header("📊 Analysis Results")
        
        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            confidence = st.session_state['confidence']
            all_predictions = st.session_state['all_predictions']
            threshold = st.session_state['threshold']
            
            # Determine if prediction is confident enough
            is_confident = (confidence / 100) >= threshold
            
            # Display prediction
            if prediction == "NORMAL":
                box_class = "normal"
                icon = "✅"
                result_text = "Normal"
            else:
                box_class = "pneumonia"
                icon = "⚠️"
                result_text = prediction.replace("_", " ").title()
            
            st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <h2>{icon} Prediction: {result_text}</h2>
                    <h3>Confidence: {confidence:.2f}%</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence warning
            if not is_confident:
                st.warning(f"⚠️ Low confidence prediction (below {threshold*100:.0f}% threshold). Please consult a medical professional.")
            
            # Show all class probabilities
            st.subheader("📈 Class Probabilities")
            class_names = config['data']['classes']
            
            for i, class_name in enumerate(class_names):
                prob = all_predictions[i] * 100
                st.progress(prob / 100)
                st.write(f"**{class_name.replace('_', ' ').title()}:** {prob:.2f}%")
            
            # Disclaimer
            st.markdown("---")
            st.warning("""
                **⚠️ Medical Disclaimer:**
                This tool is for educational and research purposes only. 
                It should NOT be used as a substitute for professional medical advice, 
                diagnosis, or treatment. Always consult with a qualified healthcare provider.
            """)
        else:
            st.info("👆 Upload an X-ray image and click 'Analyze X-Ray' to see results")
            
            # Show example
            st.subheader("📸 Example X-Ray Images")
            st.write("You can test the system with sample chest X-ray images from medical databases.")

if __name__ == "__main__":
    main()

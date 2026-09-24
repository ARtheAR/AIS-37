import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Midwest Pest Risk Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS for Advanced Theming (Matches UI Mockup)
# --------------------------------------------------
st.markdown("""
<style>
/* Main app background and text colors */
.stApp { 
    background-color: #151821; 
    color: #E0E0E0; 
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Hide Streamlit default header and footer for a cleaner dashboard look */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Adjust block container padding */
.block-container { 
    padding-top: 1.5rem; 
    padding-bottom: 2rem; 
}

/* Style headings */
h1, h2, h3, h4, h5 { 
    color: #FFFFFF; 
}

/* Main column card styling */
div[data-testid="column"] {
    background-color: #1E222D;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #2A2E3D;
}

/* Specifically target the Center Column to create the glowing border effect from the mockup */
div[data-testid="column"]:nth-of-type(2) {
    border: 2px solid #21D375;
    background-color: #1A2027;
    box-shadow: 0 0 20px rgba(33, 211, 117, 0.15);
}

/* Standard Button Styling */
.stButton > button {
    background-color: #2B303F;
    color: #FFFFFF;
    border: 1px solid #404658;
    border-radius: 6px;
    font-weight: bold;
}
.stButton > button:hover {
    border-color: #21D375;
    color: #21D375;
}

/* Primary Button Styling (For Execute Strategy) */
button[kind="primary"] {
    background-color: #21D375 !important;
    color: #151821 !important;
    border: none !important;
}
button[kind="primary"]:hover {
    background-color: #1BAF61 !important;
    color: #151821 !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Model Configuration & Data Constants
# --------------------------------------------------
MODEL_PATH = "plant_disease_model.keras"
IMG_SIZE = (224, 224)

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

risk_labels = {
    1: "No Imminent Risk",
    2: "Minimal Risk",
    3: "Moderate Risk",
    4: "High Risk",
    5: "Severe Risk"
}

# --------------------------------------------------
# Data & Model Loaders
# --------------------------------------------------
@st.cache_resource
def load_model():
    """Loads the Keras model gracefully if it exists."""
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        return None

loaded_model = load_model()

@st.cache_data
def load_risk_scores():
    """Loads JSON pest data with a robust fallback to ensure it runs universally."""
    primary_path = r'C:\Users\aayus\Hackathon\risk_scores.json'
    fallback_path = 'risk_scores.json'
    
    try:
        with open(primary_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        try:
            with open(fallback_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            st.error("Error: 'risk_scores.json' not found in absolute or local path.")
            return {}

risk_data = load_risk_scores()

# --------------------------------------------------
# Prediction Function (Preserved Original Logic)
# --------------------------------------------------
def predict_uploaded_image(uploaded_file):
    image = tf.keras.utils.load_img(
        uploaded_file,
        target_size=IMG_SIZE
    )
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(
        image_array,
        axis=0
    )
    predictions = loaded_model.predict(
        image_array,
        verbose=0
    )
    predicted_index = np.argmax(
        predictions[0]
    )
    predicted_class = class_names[
        predicted_index
    ]
    confidence = (
        predictions[0][predicted_index] * 100
    )
    return image, predicted_class, confidence


# --------------------------------------------------
# Left Sidebar Navigation (UI aesthetic match)
# --------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem; margin-top: 1rem;'>
        <span style='color: #21D375; font-size: 3rem;'>🌿</span>
    </div>
    """, unsafe_allow_html=True)
    
    nav_items = [
        ("Summary", True), 
        ("Fields", False), 
        ("Diagnostics", False), 
        ("Predictions", False), 
        ("Reports", False), 
        ("Settings", False)
    ]
    
    for item, is_active in nav_items:
        if is_active:
            st.markdown(f"""
            <div style='background-color: #2A2E3D; padding: 12px; border-radius: 8px; color: #21D375; font-weight: bold; text-align: center; margin-bottom: 10px; box-shadow: inset 4px 0 0 #21D375;'>
                {item}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='padding: 12px; color: #888; text-align: center; margin-bottom: 10px; font-weight: 500;'>
                {item}
            </div>
            """, unsafe_allow_html=True)


# --------------------------------------------------
# Main Dashboard Layout
# --------------------------------------------------
st.markdown("<h2 style='margin-top: -2.5rem; margin-bottom: 1.5rem;'>Dashboard</h2>", unsafe_allow_html=True)

# Emulate the 3-column layout proportions seen in the image
col1, col2, col3 = st.columns([1.2, 1.8, 1.1], gap="medium")

# --------------------------------------------------
# Column 1: Field Diagnostic & Image Scanner
# --------------------------------------------------
with col1:
    st.markdown("<h4 style='margin-bottom: 0px;'>Field Diagnostic</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888; font-size:0.85em; margin-bottom: 15px;'>Field: East Orchard - Block A - Current</p>", unsafe_allow_html=True)
    
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_file = st.file_uploader(
        "Upload Plant Image",
        type=["jpg", "jpeg", "png"],
        key=f"plant_image_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        if loaded_model is not None:
            image, predicted_class, confidence = predict_uploaded_image(uploaded_file)
            st.image(image, use_container_width=True)
            
            # Mimic the diagnostic bounding box/overlay text below the image
            st.markdown(f"""
            <div style='background-color:#2A2E3D; padding:12px; border-radius:6px; margin-top:15px; border-left: 4px solid #E02020;'>
                <strong style='color:#FFF; font-size:1.05em;'>DISEASE DIAGNOSIS: {predicted_class.upper()}</strong><br/>
                <span style='color:#21D375; font-weight:bold;'>(CONFIDENCE {confidence:.0f}%)</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Model unavailable. Missing 'plant_disease_model.keras'.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("RESCAN FIELD BLOCK", use_container_width=True):
        st.session_state.uploader_key += 1
        st.rerun()

# --------------------------------------------------
# Column 2: Glowing Action Plan Card
# --------------------------------------------------
with col2:
    st.markdown("<h4 style='margin-bottom: 15px;'>Action Plan Card</h4>", unsafe_allow_html=True)
    
    # Internal HTML structure for the Action Plan details
    st.markdown("""
    <div style='display: flex; justify-content: space-between; align-items: baseline;'>
        <h5 style='color:#21D375; font-weight:bold; margin: 0;'>[URGENT ACTION PLAN: BIOCONTROL INITIATION]</h5>
        <span style='color:#888; font-size:0.85em;'>Issued: Active</span>
    </div>
    <hr style='border-color: #333847; margin-top: 10px; margin-bottom: 20px;'>
    """, unsafe_allow_html=True)
    
    # Populate actual proposals directly from the risk_scores dataset
    if risk_data:
        for pest, data in risk_data.items():
            proposal = data['proposal']
            
            # If recommendation says "Do Not Spray" or "BLOCK", status is NOT green.
            if "do not spray" in proposal.lower() or "block" in proposal.lower():
                indicator_color = "#E02020" # Red for Blocked/Warning
                status_text = "[SPRAY BLOCKED]"
            else:
                indicator_color = "#21D375" # Green for Actionable
                status_text = "[ACTION RECOMMENDED]"

            st.markdown(f"""
            <div style="background-color:#1E222D; padding:15px; border-radius:8px; margin-bottom:12px; border-left: 4px solid {indicator_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <h6 style='color:#FFF; margin: 0; font-size: 1em;'>[TARGET PEST: {pest.upper()}]</h6>
                    <span style='color:{indicator_color}; font-size: 0.8em; font-weight: bold;'>{status_text}</span>
                </div>
                <p style='color:#B0B5C1; font-size: 0.9em; margin: 0;'>{proposal}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No pest risk proposals available.")

    st.markdown("<br>", unsafe_allow_html=True)
    # Native Streamlit primary button mapped to the green CSS
    st.button("EXECUTE BIO-STRATEGY NOW", type="primary", use_container_width=True)

# --------------------------------------------------
# Column 3: Weather & Pest Risk Score Overview
# --------------------------------------------------
with col3:
    st.markdown("<h5 style='margin-bottom: 15px; color: #FFF; font-size: 0.95rem;'>[7-DAY WEATHER FORECAST]</h5>", unsafe_allow_html=True)

    weather_forecasts = [
        {"day": "Today", "icon": "🌧️", "temp": "25°C", "wind": "18 mph SE", "precip": "85%"},
        {"day": "Tomorrow", "icon": "☁️", "temp": "20°C", "wind": "12 mph E", "precip": "20%"},
        {"day": "Day 3", "icon": "🌧️", "temp": "21°C", "wind": "15 mph SE", "precip": "70%"},
    ]

    for w in weather_forecasts:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; background-color: #1E222D; padding: 10px; border-radius: 6px; margin-bottom: 8px; border: 1px solid #2A2E3D;">
            <div style="color: #FFF; width: 25%; font-weight: bold; font-size: 0.9em;">{w['day']}</div>
            <div style="font-size: 1.2em; width: 15%; text-align: center;">{w['icon']}</div>
            <div style="color: #21D375; width: 20%; font-size: 0.95em; text-align: center; font-weight: bold;">{w['temp']}</div>
            <div style="color: #B0B5C1; width: 40%; font-size: 0.8em; text-align: right;">{w['precip']} Rain<br>{w['wind']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h5 style='margin-bottom: 15px; color: #FFF; font-size: 0.95rem;'>[PEST RISK SCORE OVERVIEW]</h5>", unsafe_allow_html=True)
    
    if risk_data:
        for pest, data in risk_data.items():
            score = data["score"]
            label = risk_labels.get(score, 'Unknown')
            
            # Determine UI color based on risk severity score
            if score <= 2:
                color = "#21D375" # Neon Green
            elif score == 3:
                color = "#F5A623" # Warning Orange/Yellow
            else:
                color = "#E02020" # Severe Red
                
            width_pct = (score / 5.0) * 100
            
            # Draw sleek custom progress bars resembling the vertical metrics from the design
            st.markdown(f"""
            <div style="margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85em; font-weight: bold; margin-bottom: 0.4rem;">
                    <span style="color: #FFF;">{pest.upper()}</span>
                    <span style="color: {color};">{score}/5 ({label.upper()})</span>
                </div>
                <div style="background-color: #2A2E3D; border-radius: 10px; height: 12px; width: 100%; overflow: hidden;">
                    <div style="background-color: {color}; height: 100%; width: {width_pct}%; border-radius: 10px; box-shadow: 0 0 8px {color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color:#2A2E3D; padding:15px; border-radius:8px; margin-bottom:12px; border-left: 4px solid #F5A623;'>
        <div style="display: flex; align-items: center;">
            <div style="font-size: 2rem; margin-right: 15px;">🌧️</div>
            <div>
                <strong style='color:#FFF; font-size:0.95em;'>WARNING: RAIN IMMINENT</strong><br/>
                <span style='color:#B0B5C1; font-size:0.85em;'>High precipitation detected. Foliar sprays blocked to prevent runoff.</span>
            </div>
        </div>
    </div>
    <div style='background-color:#2A2E3D; padding:15px; border-radius:8px; border-left: 4px solid #F5A623;'>
        <div style="display: flex; align-items: center;">
            <div style="font-size: 2rem; margin-right: 15px;">💨</div>
            <div>
                <strong style='color:#FFF; font-size:0.95em;'>WARNING: HIGH WIND</strong><br/>
                <span style='color:#B0B5C1; font-size:0.85em;'>Wind speeds elevated. Reduce spray application height if manual treatment used.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
import streamlit as st
import tensorflow as tf
import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Disease Diagnosis",
    page_icon="🌿",
    layout="wide"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "plant_disease_model.keras"

IMG_SIZE = (224, 224)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    'Apple___Apple_scab',
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
    'Tomato___healthy'
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


loaded_model = load_model()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_uploaded_image(uploaded_file):

    # Same preprocessing used in the original working model code
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


# ============================================================
# PAGE TITLE
# ============================================================

st.title("Plant Disease Diagnostic System")

st.write(
    "Upload a plant image to detect possible disease."
)


# ============================================================
# RESCAN BUTTON
# ============================================================

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


if st.button(
    "RESCAN FIELD BLOCK",
    use_container_width=True
):

    st.session_state.uploader_key += 1

    st.rerun()


# ============================================================
# IMAGE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Plant Image",
    type=["jpg", "jpeg", "png"],
    key=f"plant_image_{st.session_state.uploader_key}"
)


# ============================================================
# IMAGE PREDICTION AND DISPLAY
# ============================================================

if uploaded_file is not None:

    image, predicted_class, confidence = (
        predict_uploaded_image(uploaded_file)
    )

    st.image(
        image,
        caption="Uploaded Plant Image",
        use_container_width=True
    )

    st.subheader("Disease Diagnosis")

    st.write(
        predicted_class
    )

    st.subheader("Confidence")

    st.write(
        f"{confidence:.2f}%"
    )
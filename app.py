
import streamlit as st
import tensorflow as tf
import numpy as np
import json

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# 1. Application configuration
# -----------------------------

st.set_page_config(
    page_title="Mango Quality AI",
    page_icon="🥭",
    layout="centered"
)

st.title("🥭 Mango Quality Classification")
st.write(
    "Upload a mango image and let AI classify its quality."
)

# -----------------------------
# 2. Load model configuration
# -----------------------------

with open("model_config.json", "r") as f:
    config = json.load(f)

CLASS_NAMES = config["class_names"]
IMG_SIZE = config["image_size"]
PREPROCESSING = config["preprocessing"]

# -----------------------------
# 3. Load trained model
# -----------------------------

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "final_mango_quality_model.keras",
        compile=False
    )

model = load_model()

# -----------------------------
# 4. Image upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Mango Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Mango Image",
        use_container_width=True
    )

    if st.button("Predict Mango Quality"):

        with st.spinner("AI is analyzing your mango..."):

            # Resize image
            image_resized = image.resize(
                (IMG_SIZE, IMG_SIZE)
            )

            # Convert to numerical array
            image_array = np.array(
                image_resized,
                dtype=np.float32
            )

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Apply model-specific preprocessing
            if PREPROCESSING == "mobilenet_v2":

                image_array = preprocess_input(
                    image_array
                )

            else:

                image_array = image_array / 255.0

            # Predict
            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            predicted_index = np.argmax(predictions)

            predicted_class = CLASS_NAMES[
                predicted_index
            ]

            confidence = predictions[
                predicted_index
            ] * 100

        # -----------------------------
        # 5. Display prediction
        # -----------------------------

        st.subheader("Prediction Result")

        st.success(
            f"Predicted Class: {predicted_class}"
        )

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        # -----------------------------
        # 6. Display probabilities
        # -----------------------------

        st.subheader("Class Probabilities")

        probability_data = {
            "Mango Class": CLASS_NAMES,
            "Probability (%)": (
                predictions * 100
            ).round(2)
        }

        st.dataframe(
            probability_data,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            {
                name: float(value * 100)
                for name, value in zip(
                    CLASS_NAMES,
                    predictions
                )
            }
        )

        st.caption(
            "Prediction confidence is the model's "
            "estimated class probability, not a guarantee "
            "of actual mango quality."
        )

st.divider()

st.caption(
    "Mango Quality Classification | Deep Learning Project"
)

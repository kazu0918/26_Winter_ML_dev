from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pickle

import streamlit as st
from streamlit_mnist_canvas import st_mnist_canvas

import numpy as np
import pandas as pd
import altair as alt

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="MNIST Digit Classifier Demo", layout="wide")


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "mnist_mlp.pkl"
MNIST_IMAGE_PATH = BASE_DIR / "MNIST_dataset_example.png"


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


clf = load_model()


# -----------------------------
# Preprocess image
# -----------------------------
def preprocess_image(img):
    """
    img: 28x28 grayscale image
    return: shape (1, 784)
    """

    x = np.asarray(img).astype("float32")

    # 0〜255 の場合は 0〜1 に変換
    if x.max() > 1.0:
        x = x / 255.0

    # MNIST is usually black background + white digit.
    # If predictions look strange, try changing this to True.
    INVERT_IMAGE = False

    if INVERT_IMAGE:
        x = 1.0 - x

    x = x.reshape(1, -1)

    return x


# -----------------------------
# Prediction function
# -----------------------------
def predict_digit(img):
    x = preprocess_image(img)

    pred_proba = clf.predict_proba(x)[0]

    probs = np.zeros(10)

    for cls, prob in zip(clf.classes_, pred_proba):
        probs[int(cls)] = prob

    pred_digit = int(np.argmax(probs))

    return pred_digit, probs


# -----------------------------
# Session state
# -----------------------------
if "results" not in st.session_state:
    st.session_state.results = []


# -----------------------------
# Main centered layout
# -----------------------------
page_left, page_main, page_right = st.columns([1, 3, 1])

with page_main:

    # -----------------------------
    # Header
    # -----------------------------
    st.title("MNIST Digit Classifier Demo")

    st.markdown(
        """
<div style="font-size: 18px; line-height: 1.7; margin-top: 10px; margin-bottom: 20px;">

<p>
This app is a simple demo of handwritten digit classification using the
<strong>MNIST dataset</strong>.
</p>

<p>
In this demo app, the prediction model is implemented using
<strong>scikit-learn's MLPClassifier</strong> for simplicity.
However, in the workshop, we will build the backend neural network model
<strong>from scratch using NumPy</strong>, so that we can understand what happens inside the model.
</p>

<p>
Draw one handwritten digit from <strong>0 to 9</strong> in the input canvas below.
The trained model will predict which digit you wrote and show the probability for each digit.
</p>

</div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------
    # MNIST image
    # -----------------------------
    if MNIST_IMAGE_PATH.exists():
        image_left, image_center, image_right = st.columns([1, 2, 1])

        with image_center:
            st.image(
                MNIST_IMAGE_PATH,
                caption="Example images from the MNIST dataset",
                use_container_width=True,
            )

    st.markdown(
        """
<div style="font-size: 15px; margin-top: 5px; margin-bottom: 35px;">
Learn more about the
<a href="https://en.wikipedia.org/wiki/MNIST_database" target="_blank">
MNIST database on Wikipedia</a>.
</div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------
    # Input
    # -----------------------------
    st.subheader("Input")

    st.markdown("""
Please draw **one digit from 0 to 9** in the canvas below, then submit it.
        """)

    canvas_left, canvas_center, canvas_right = st.columns([1, 2, 1])

    with canvas_center:
        result = st_mnist_canvas()

    # -----------------------------
    # Submit
    # -----------------------------
    if result.is_submitted:
        img = result.resized_grayscale_array

        if img is not None:
            pred_digit, probs = predict_digit(img)

            predicted_at = datetime.now(ZoneInfo("Australia/Perth")).strftime(
                "%H:%M:%S"
            )

            st.session_state.results.append(
                {
                    "image": img,
                    "prediction": pred_digit,
                    "probs": probs,
                    "predicted_at": predicted_at,
                }
            )

    # -----------------------------
    # Results
    # -----------------------------
    st.subheader("Results")

    st.markdown("""
After submission, the result will show:

- the input image resized to **28 × 28 pixels**
- the model's predicted digit
- the predicted probability for each digit from **0 to 9**
        """)

    if st.button("Clear results"):
        st.session_state.results = []

    if len(st.session_state.results) == 0:
        st.write("No results yet.")

    else:
        for i, item in enumerate(reversed(st.session_state.results), start=1):
            is_latest = i == 1

            if is_latest:
                st.markdown(f"### Latest Result — predicted at {item['predicted_at']}")
            else:
                st.markdown(f"### Result {i} — predicted at {item['predicted_at']}")

            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                st.write("Input")
                st.image(item["image"], caption="28 × 28", width=90)

            with col2:
                st.write("Prediction")

                font_size = 80 if is_latest else 42

                st.markdown(
                    f"""
<h1 style="
    text-align: center;
    margin-top: 20px;
    font-size: {font_size}px;
">
{item["prediction"]}
</h1>
                    """,
                    unsafe_allow_html=True,
                )

            with col3:
                st.write("Probability")

                prob_df = pd.DataFrame(
                    {
                        "digit": [str(d) for d in range(10)],
                        "probability": item["probs"],
                    }
                )

                chart = (
                    alt.Chart(prob_df)
                    .mark_bar()
                    .encode(
                        y=alt.Y(
                            "digit:N",
                            sort=[str(d) for d in range(10)],
                            title="digit",
                        ),
                        x=alt.X(
                            "probability:Q",
                            scale=alt.Scale(domain=[0, 1]),
                            title="probability",
                        ),
                    )
                    .properties(height=220)
                )

                st.altair_chart(chart, use_container_width=True)

            st.divider()

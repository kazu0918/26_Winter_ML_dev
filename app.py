import streamlit as st
from streamlit_mnist_canvas import st_mnist_canvas
import numpy as np
import pandas as pd
import altair as alt
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="MNIST Canvas Test", layout="wide")

st.title("MNIST Canvas Test")

# -----------------------------
# 結果履歴を保存する
# -----------------------------
if "results" not in st.session_state:
    st.session_state.results = []


# -----------------------------
# 仮の予測関数
# 後で本物のモデルに置き換える
# -----------------------------
def predict_digit(img):
    probs = np.random.random(10)
    probs = probs / probs.sum()

    pred_digit = int(np.argmax(probs))

    return pred_digit, probs


# -----------------------------
# Input canvas を中央に配置
# centerを狭めにする
# -----------------------------
input_left, input_center, input_right = st.columns([1, 5, 1])

with input_center:
    st.subheader("Input")
    result = st_mnist_canvas()


# -----------------------------
# submitされたら結果に追加
# -----------------------------
if result.is_submitted:
    img = result.resized_grayscale_array
    pred_digit, probs = predict_digit(img)

    predicted_at = datetime.now(ZoneInfo("Australia/Perth")).strftime("%H:%M:%S")

    st.session_state.results.append({
        "image": img,
        "prediction": pred_digit,
        "probs": probs,
        "predicted_at": predicted_at
    })


# -----------------------------
# Results section
# ここは左右に余白を取る
# -----------------------------
results_left, results_main, results_right = st.columns([1, 5, 1])

with results_main:
    st.subheader("Results")

    if st.button("Clear results"):
        st.session_state.results = []

    if len(st.session_state.results) == 0:
        st.write("No results yet.")

    else:
        # 新しい結果を上に表示
        for i, item in enumerate(reversed(st.session_state.results), start=1):

            is_latest = i == 1

            if is_latest:
                st.markdown(f"### Latest Result — predicted at {item['predicted_at']}")
            else:
                st.markdown(f"### Result {i} — predicted at {item['predicted_at']}")

            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                st.write("Input")
                st.image(
                    item["image"],
                    caption="28x28",
                    width=80
                )

            with col2:
                st.write("Prediction")

                font_size = 80 if is_latest else 42

                st.markdown(
                    f"""
                    <h1 style='
                        text-align: center;
                        margin-top: 20px;
                        font-size: {font_size}px;
                    '>
                        {item["prediction"]}
                    </h1>
                    """,
                    unsafe_allow_html=True
                )

            with col3:
                st.write("Probability")

                prob_df = pd.DataFrame({
                    "digit": [str(d) for d in range(10)],
                    "probability": item["probs"]
                })

                chart = (
                    alt.Chart(prob_df)
                    .mark_bar()
                    .encode(
                        y=alt.Y(
                            "digit:N",
                            sort=[str(d) for d in range(10)],
                            title="digit"
                        ),
                        x=alt.X(
                            "probability:Q",
                            scale=alt.Scale(domain=[0, 1]),
                            title="probability"
                        )
                    )
                    .properties(
                        height=220
                    )
                )

                st.altair_chart(chart, use_container_width=True)

            st.divider()
import streamlit as st
from streamlit_mnist_canvas import st_mnist_canvas
import numpy as np

st.title("MNIST Canvas Test")

st.title("はじめてのStreamlit")
st.header("見出し（h2相当）")
st.subheader("サブ見出し（h3相当）")
st.markdown("""
**Markdown**も使えます。  
- 箇条書き
- *斜体* や **太字**
- コード: `st.write("hello")`
""")
st.code('print("コードブロックもOK")', language="python")
st.write({"辞書やDataFrame": "そのまま表示可"})

# サイドバー
st.sidebar.title("コントロール")
name = st.sidebar.text_input("お名前", value="ゲスト")
level = st.sidebar.slider("難易度", 1, 10, 3)
agree = st.sidebar.checkbox("規約に同意する", value=True)

col1, col2 = st.columns(2)
with col1:
    st.write(f"こんにちは、**{name}** さん！")
with col2:
    if agree:
        st.success(f"難易度は {level} に設定されました。")
    else:
        st.warning("同意が必要です。")

tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
tab1.write("this is tab 1")
tab2.write("this is tab 2")

st.subheader("Input")
result = st_mnist_canvas()

if result.is_submitted:
    st.write("Output")

    img = result.resized_grayscale_array

    st.image(img, caption="Grayscale 28x28 Image")

    # st.write("Array shape:", img.shape)
    # st.write("Array type:", type(img))
    # st.write("Min value:", np.min(img))
    # st.write("Max value:", np.max(img))

    # Prepare the image for ML model prediction
    # image_for_prediction = np.expand_dims(img, axis=0)

    # st.write("Shape for prediction:", image_for_prediction.shape)

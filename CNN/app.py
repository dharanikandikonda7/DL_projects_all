import streamlit as st
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

st.set_page_config(
    page_title="CNN Convolution Demo",
    page_icon="🖼️"
)

st.title("CNN Image Processing Demo")

st.write(
    "Upload an image and see how CNN convolution filters extract features."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("L")

    img_array = np.array(image)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    filters = {
        "Edge Detection":
        np.array([
            [-1,-1,-1],
            [-1, 8,-1],
            [-1,-1,-1]
        ]),

        "Sharpen":
        np.array([
            [0,-1,0],
            [-1,5,-1],
            [0,-1,0]
        ]),

        "Box Blur":
        np.ones((3,3))/9
    }

    selected_filter = st.selectbox(
        "Choose Filter",
        list(filters.keys())
    )

    kernel = filters[selected_filter]

    result = convolve2d(
        img_array,
        kernel,
        mode="same",
        boundary="symm"
    )

    result = np.clip(result, 0, 255)

    st.subheader("Filtered Output")
    st.image(
        result.astype(np.uint8),
        use_container_width=True
    )

    st.write("Selected Kernel")

    st.dataframe(kernel)
import streamlit as st
from PIL import Image
from ultralytics import YOLO


st.title("Bahraini Food Detection App")


@st.cache_resource
def load_model():
  return YOLO("best.pt")


try:
  model = load_model()
except Exception as e:
  st.warning(
      "Model weights ('best.pt') not found. Please place your trained model"
      " file in the app directory."
  )
  model = None

tab1, tab2 = st.tabs([" Upload Image", " Live Camera"])

with tab1:
  st.subheader("Upload an Image")
  uploaded_file = st.file_uploader(
      "Choose an image...", type=["jpg", "jpeg", "png"]
  )

  if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
      st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
      with st.spinner("Analyzing image..."):
        results = model(image)
        res_image = results[0].plot()[..., ::-1]
        st.image(
            res_image, caption="Detected Currency Notes", use_container_width=True
        )

with tab2:
  st.subheader("Take a Picture with Camera")
  camera_image = st.camera_input("Capture note")

  if camera_image is not None and model is not None:
    image = Image.open(camera_image)
    col1, col2 = st.columns(2)

    with col1:
      st.image(image, caption="Captured Photo", use_container_width=True)

    with col2:
      with st.spinner("Analyzing frame..."):
        results = model(image)
        res_image = results[0].plot()[..., ::-1]
        st.image(
            res_image, caption="Detection Result", use_container_width=True
        )

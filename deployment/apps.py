import time
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
from utils import *
import plotly.graph_objects as go

# HTML header
html_temp = '''
    <div style="padding-bottom: 10px; padding-top: 20px; padding-left: 5px; padding-right: 5px">
    <center><h1>Pneumonia Disease Detection</h1></center>
    </div>
    '''
st.markdown(html_temp, unsafe_allow_html=True)

# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Classification", "Model Evaluation"])

# Model Selection
st.sidebar.title("Model Selection")
model_option = st.sidebar.selectbox("Select the model to use:", ('Model V1', 'Model V2'))

# Load the selected model
if model_option == 'Model V1':
    selected_model = modelV1()
    # Metrik untuk Model V1
    accuracy_v1 = 87.50
    precision_v1 = 84.21
    recall_v1 = 98.46
    f1_score_v1 = 90.78
elif model_option == 'Model V2':
    selected_model = modelV2()
    # Metrik untuk Model V2
    accuracy_v2 = 91.51
    precision_v2 = 91.00
    recall_v2 = 95.90
    f1_score_v2 = 93.38

# Classification Page
if page == "Classification":
    st.header("Upload Lung CT Scan Image for Classification")

    opt = st.selectbox("How do you want to upload the image for classification?", ('Please Select', 'Upload image via link', 'Upload image from device'))

    image = None
    if opt == 'Upload image from device':
        file = st.file_uploader('Select', type=['jpg', 'png', 'jpeg'])
        if file is not None:
            image = Image.open(file)

    elif opt == 'Upload image via link':
        img = st.text_input('Enter the Image Address')
        try:
            image = Image.open(urllib.request.urlopen(img))
        except:
            if st.button('Submit Link'):
                show = st.error("Please Enter a valid Image Address!")
                time.sleep(4)
                show.empty()

    try:
        if image is not None:
            st.image(image, width=224, caption='Uploaded Image')
            if st.button('Predict'):
                with st.spinner('Classifying... Please wait.'):
                    prepare_img = preprocess(image)
                    # Predict class
                    predicted_class, confidence = binary_predict_image(selected_model, image, threshold=0.5)
                st.info(f'Hey! The uploaded image has been classified as "{predicted_class}" with confidence "{confidence}".')

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Model Evaluation Page
elif page == "Model Evaluation":
    st.header("Model Evaluation Metrics")

    if st.button("Evaluate Model"):
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)

        if model_option == 'Model V1':
            # Tampilkan metrik evaluasi untuk Model V1
            accuracy = accuracy_v1
            precision = precision_v1
            recall = recall_v1
            f1_score = f1_score_v1
        elif model_option == 'Model V2':
            # Tampilkan metrik evaluasi untuk Model V2
            accuracy = accuracy_v2
            precision = precision_v2
            recall = recall_v2
            f1_score = f1_score_v2

        st.write(f"**Model:** {model_option}")
        st.write(f"**Accuracy**: {accuracy}%")
        st.write(f"**Precision**: {precision}%")
        st.write(f"**Recall**: {recall}%")
        st.write(f"**F1-Score**: {f1_score}%")

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy, precision, recall, f1_score]

        fig = go.Figure(data=[go.Bar(x=metrics, y=values, marker_color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA'])])

        fig.update_layout(
            title=f'Model Evaluation Metrics - {model_option}',
            xaxis_title='Metrics',
            yaxis_title='Percentage (%)',
            yaxis=dict(range=[0, 100])  # Batasi sumbu y dari 0 hingga 100
        )

        # Tampilkan grafik
        st.plotly_chart(fig)
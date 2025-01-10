# Pneumonia Disease Detection

A web application to classify chest X-ray images as either "Normal" or "Pneumonia" using convolutional neural networks (CNNs) built on the VGG19 architecture.

## Features
- **Classification**: Upload chest X-ray images via file or link for classification.
- **Model Evaluation**: View evaluation metrics such as Accuracy, Precision, Recall, and F1-Score for the available models.
- **Interactive Visualization**: Display performance metrics using bar charts.

## How It Works
1. Select the desired model (`Model V1` or `Model V2`) from the sidebar.
2. Navigate to the "Classification" page to upload an X-ray image.
3. The application will preprocess the image and predict whether it shows signs of pneumonia or is normal.
4. Navigate to the "Model Evaluation" page to view the performance metrics of the selected model.

## Models
The application uses two VGG19-based models:
- `Model V1` with an accuracy of **91.83%**
- `Model V2` with an accuracy of **91.19%**

Both models are trained and saved in `.h5` format.

## Libraries Used
The following libraries are used in the application:
- `tensorflow` and `keras`: For building and loading the VGG19 models.
- `streamlit`: To create an interactive web interface.
- `numpy`: For numerical computations.
- `Pillow`: For image processing.
- `plotly`: To visualize model metrics interactively.
- `urllib`: For handling image URLs.
- `requests`: For managing external data sources.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/pneumonia-detection.git
2. Navigate to the project directory:
   ```bash
   cd pneumonia-detection
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
5. Run the application
   ```bash
   streamlit run apps.py

## Directory Structure
pneumonia-detection/
├── apps.py           # Main Streamlit application
├── utils.py          # Utility functions for preprocessing and model handling
├── model/            # Folder containing the pre-trained models
├── requirements.txt  # List of required Python libraries
└── README.md         # Project documentation

## Author
Abdullah Farauk/Stardenbart

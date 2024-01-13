import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('brain.h5')

# Class labels
class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def load_and_predict(image):
    # Preprocess the image for prediction
    image = cv2.resize(image, (150, 150))  # Resize the image to match the input shape of the model
    image = np.expand_dims(image, axis=0)  # Add an extra dimension for batch size

    # Make predictions
    predictions = model.predict(image)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_idx]

    return predicted_class

def main():
    st.title("Brain Tumor Classifier")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        if st.button("Predict"):
            predicted_class = load_and_predict(image)
            st.success(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()

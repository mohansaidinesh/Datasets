import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3
import tempfile
import os
# Load the models
digit_model = load_model("digitSignLanguage.h5")
alphabet_model = load_model("indianSignLanguage.h5")
# Mapping for alphabet predictions
alphabet_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
                    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Image file names
digit_image_files = [f"{i}.jpeg" for i in range(10)]
alphabet_image_files = [f"{chr(65 + i)}.jpg" for i in range(26)]

# Streamlit UI
st.title("Sign Language Detection")

# Navigation menu
navigation_menu = st.sidebar.radio("Home", ["View Signs","Digits", "Alphabets"])

# Function to predict and display results
def predict_and_display(model, image, mapping):
    if isinstance(image, Image.Image):
        # If it's an Image object, continue with the existing logic
        # Ensure the image has 3 color channels (RGB)
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Preprocess the image
        resized_image = image.resize((32, 32))
        input_image = np.array(resized_image) / 255.0
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
        # Make predictions
        predictions = model.predict(input_image)
        predicted_class_index = np.argmax(predictions)
        # Display the results
        st.image(image, caption=f"Predicted class: {mapping[predicted_class_index]}", use_column_width=True)
        st.write(f"Predicted class: {mapping[predicted_class_index]}")

        # Convert the predicted class to speech and save to a temporary file
        speak_output = f"The predicted class is {mapping[predicted_class_index]}"
        with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
            temp_audio_path = temp_audio.name + ".wav"
            engine = pyttsx3.init()
            engine.save_to_file(speak_output, temp_audio_path)
            engine.runAndWait()

        # Play the saved audio file
        st.audio(temp_audio_path, format="audio/wav", start_time=0)
        os.remove(temp_audio_path)  # Remove the temporary audio file after playing
    else:
        # If it's an UploadedFile object, read the content as an image
        image = Image.open(image)
        predict_and_display(model, image, mapping)

# Digit detection
if navigation_menu == "Digits":
    st.header("Digit Sign Language Detection")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        predict_and_display(digit_model, image, mapping={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'})
# Alphabet detection
elif navigation_menu == "Alphabets":
    st.header("Alphabet Sign Language Detection")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        predict_and_display(alphabet_model, image, mapping=alphabet_mapping)

# View Images
elif navigation_menu == "View Signs":
    st.subheader("Digits:")
    digit_images = [Image.open(file) for file in digit_image_files]
    st.image(digit_images, caption=[str(i) for i in range(10)], width=100)

    st.subheader("Alphabets:")
    alphabet_images = [Image.open(file) for file in alphabet_image_files]
    st.image(alphabet_images, caption=[chr(65 + i) for i in range(26)], width=100)

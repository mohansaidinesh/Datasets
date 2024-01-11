import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the glaucoma detection model
glaucoma_model = tf.keras.models.load_model('my_model2.h5')

# Load the diabetic retinopathy detection model
retinopathy_model = load_model('model.hd5')
image_size = (64, 64)

def preprocess_image(image, target_size):
    # Convert the image to the required size and format
    img = image.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def glaucoma_detection(image):
    image = ImageOps.fit(image, (100, 100), Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis, ...]
    prediction = glaucoma_model.predict(img_reshape)
    return prediction[0][0]

def retinopathy_detection(image):
    processed_image = preprocess_image(image, image_size)
    prediction = retinopathy_model.predict(processed_image)
    return prediction[0][0]

def main():
    st.title("Disease Detection App")

    st.header("Upload an eye image for disease detection.")
    file = st.file_uploader("Please upload an image (jpg) file", type=["jpg"])

    if file is not None:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", width=300)

        # Buttons for disease detection
        if st.button("Glaucoma Detection"):
            # Glaucoma detection
            prediction = glaucoma_detection(image)
            if prediction > 0.5:
                st.subheader("Prediction Result:")
                st.success("Healthy eyes detected!")
            else:
                st.subheader("Prediction Result:")
                st.warning("Glaucoma detected! Please consult an ophthalmologist.")

        if st.button("Diabetic Retinopathy Detection"):
            # Diabetic Retinopathy detection
            processed_image = preprocess_image(image, image_size)
            prediction = retinopathy_model.predict(processed_image)
            st.subheader("Prediction Result:")
            if prediction > 0.4:
                st.success("Diabetic Retinopathy Detected!")
            else:
                st.success("No Diabetic Retinopathy Detected!")

if __name__ == "__main__":
    main()
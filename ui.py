import streamlit as st
import os
import io
import cv2
import mediapipe as mp
import numpy as np
import pickle
import warnings
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set up environment variables
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = 'sk-mjO2T3pxCHJMzG7L25kEDmWYBlts6hDO1yP5wDDwLXhEC4Nh'

# Fixed parameters
seed = 4253978046
steps = 50
cfg_scale = 8.0
width = 1024
height = 1024
samples = 5
sampler = "k_dpmpp_2m"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
max_length = 25
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Function to generate images
def generate_images(prompt):
    stability_api = client.StabilityInference(
        key=os.environ['STABILITY_KEY'],
        verbose=True,
        engine="stable-diffusion-xl-1024-v1-0",
    )
    answers = stability_api.generate(
        prompt=prompt,
        seed=seed,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        samples=samples,
        sampler=getattr(generation, "SAMPLER_" + sampler.upper())
    )

    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                img.save(str(artifact.seed) + ".png")
                st.image(img, caption="Generated Image", use_column_width=True)
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds
def image_processed(hand_img):
    # Image processing
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    img_flip = cv2.flip(img_rgb, 1)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
    output = hands.process(img_flip)
    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        data = str(data)
        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']
        without_garbage = [i for i in data if i not in garbage]

        clean = [float(i.strip()[2:]) for i in without_garbage]

        return clean
    except:
        return np.zeros([1,63], dtype=int)[0]
# Streamlit UI
st.title("Image Generation App")
page = st.sidebar.radio("Navigation", ["Home", "Txt2Img", "Img2Txt","Sign Language"])
if page == "Home":
    st.image('1.jpeg',  use_column_width=True)
elif page == "Txt2Img":
    prompt = st.text_input("Enter prompt")
    generate_images(prompt)
elif page == "Img2Txt":
    image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if image:
        st.image(image, caption="Uploaded Image", width=200)
        preds = predict_step([image])
        st.write("Predicted Text:", preds[0])
elif page == "Sign Language":
    # Load model
    with open('model.pkl', 'rb') as f:
        svm = pickle.load(f)

    # Webcam capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot open camera")
    else:
        st.success("Camera is ready!")

    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("Can't receive frame (stream end?). Exiting ...")
            break

        data = image_processed(frame)
        data = np.array(data)
        y_pred = svm.predict(data.reshape(-1, 63))

        # Using cv2.putText() method
        frame = cv2.putText(frame, str(y_pred[0]), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5, cv2.LINE_AA)
        
        # Convert the frame to an image that Streamlit can display.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, channels="RGB", use_column_width=True)
        
        # Clear the Streamlit cache to display only the latest frame
        st.experimental_rerun()

    cap.release()
    cv2.destroyAllWindows()


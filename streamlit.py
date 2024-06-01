import streamlit as st
import base64
import tempfile
import os
from PIL import Image
from io import BytesIO
import streamlit as st
import torch
import glob
import numpy as np
import os
import cv2
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import time
import smtplib
import tempfile
import zipfile
import json
import pandas as pd
import altair as alt
from model_g import generate_model
from model_f import Learner
from modell import ResNeXtBottleneck


# Define function to set the video as the page background
def set_video_as_page_bg(video_file):
    # Read the video file
    video_data = open(video_file, "rb").read()

    # Convert video to base64
    video_base64 = base64.b64encode(video_data).decode()

    # Set the background video using HTML5 video tag with controlled size
    page_bg_video = f'''
    <style>
    .stApp {{
        background: none;
    }}
    #bg-video {{
        position: fixed;
        top: 100%;
        left: 100%;
        width: 100%; /* Control the width of the video */
        height: auto; /* Maintain aspect ratio */
        transform: translate(-100%, -100%);
        z-index: -1;
    }}
    </style>
    <video autoplay muted loop id="bg-video">
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
    '''
    
    st.markdown(page_bg_video, unsafe_allow_html=True)

# Set the background video
set_video_as_page_bg("/Users/sarh/Documents/SDA/deep_learning/Deployment/WhatsApp Video 2024-05-30 at 2.09.02 PM.mp4")

# Define function to resize image
def resize_image(image_path, new_width, new_height):
    img = Image.open(image_path)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    return img

# Define function to convert image to base64
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Resize and convert images to base64
sidebar_image1 = resize_image(r"/Users/sarh/Documents/SDA/deep_learning/Deployment/‏لقطة الشاشة ١٤٤٥-١١-٢٠ في ١١.٠٢.٥١ م.png", 120, 120)
sidebar_image2 = resize_image(r"/Users/sarh/Documents/SDA/deep_learning/Deployment/‏لقطة الشاشة ١٤٤٥-١١-١٥ في ١.٣٠.٤٧ م.png", 100, 40)

sidebar_image1_base64 = image_to_base64(sidebar_image1)
sidebar_image2_base64 = image_to_base64(sidebar_image2)

# Create HTML to display images side by side
sidebar_html = f"""
<div style="display: flex; justify-content: space-between;">
    <img src="data:image/png;base64,{sidebar_image1_base64}" style="margin: 10px; width: 100px; height: 60px; padding-: 15px;"/>
    <img src="data:image/png;base64,{sidebar_image2_base64}" style="margin: 10px; width: 120px; height:60px;margin-right:600px; padding-top: 20px; padding-left: 20px;"/>
</div>
"""

# Sidebar content
st.sidebar.markdown(sidebar_html, unsafe_allow_html=True)

# Main content
st.markdown('<h1 style="color:#8DAD73;text-align:center;">رَقــــــــــيــب</h1>', unsafe_allow_html=True)
# Load location data from JSON file
# Load location data from JSON file
with open('/Users/sarh/Documents/SDA/deep_learning/Deployment/location.json', 'r') as f:
    location_data=json.load(f)
# Define function to load models
def load_models():
    model = generate_model()  # feature extractor
    classifier = Learner()  # classifier
    model.eval()
    classifier.eval()
    return model, classifier
model, classifier = load_models()
# Define function to send alert
def send_alert(subject, body):
    # Implement your email sending functionality here
    print(subject)
    print(body)

# Define transformation
transform = transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.ToTensor(),
])

abnormal_threshold = 0.45

def display_location(file_name):
    name = os.path.splitext(file_name)[0]
    location = location_data.get(name, None)
    if location:
        st.write(f"Location: {location['City']}, {location['Street']}")
    else:
        st.write("Location data not available for this video.")
# Define function to process video
def process_video(video_file):
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    # Directory containing the images
    save_path = os.path.join("/tmp", os.path.basename(tfile.name) + '_result')
    os.makedirs(save_path, exist_ok=True)

    # Prepare for video writing
    output_video_path = os.path.join("/tmp", os.path.basename(tfile.name) + '_output.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (320, 240))

    abnormal_count = 0

    # Initialize the input tensor
    inputs = torch.zeros((1, 3, 16, 240, 320))  # (batch_size, channels, num_frames, height, width)

    # Process video frames
    cap = cv2.VideoCapture(tfile.name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        if frame_idx < 16:
            inputs[0, :, frame_idx, :, :] = img  # Fill the frames initially
        else:
            inputs[0, :, :15, :, :] = inputs[0, :, 1:, :, :]  # Shift frames to the left
            inputs[0, :, 15, :, :] = img  # Add the new frame

            with torch.no_grad():
                start = time.time()
                output, feature = model(inputs)
                feature = F.normalize(feature, p=2, dim=1)
                out = classifier(feature)
                end = time.time()

            FPS = str(1 / (end - start))[:5]
            out_str = str(out.item())[:5]

            h, w, _ = frame.shape
            frame = cv2.putText(frame, f'FPS: {FPS} Pred: {out_str}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 240), 2)

            if out.item() > abnormal_threshold:
                abnormal_count += 1
                frame = cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 3)
                alert_subject = "Abnormal Activity Detected"
                alert_body = f"Abnormal activity detected at frame {frame_idx}."
                send_alert(alert_subject, alert_body)

            save_img_path = os.path.join(save_path, f"{frame_idx:04d}.jpg")
            cv2.imwrite(save_img_path, frame)
            video_writer.write(frame)

        frame_idx += 1

    cap.release()
    video_writer.release()
    # Calculate the percentage of abnormal frames
    abnormal_percentage = (abnormal_count / frame_count) * 100

    # Decide overall classification based on the abnormal percentage
    overall_classification = 'Abnormal' if abnormal_percentage >= 45 else 'Normal'

    return output_video_path, overall_classification, abnormal_percentage

def process_zip(zip_file):
      
    #Save uploaded zip to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(zip_file.read())

    with zipfile.ZipFile(tfile.name, 'r') as zip_ref:
        extract_path = os.path.join("/tmp", os.path.basename(tfile.name))
        zip_ref.extractall(extract_path)

    # Initialize the list of images
    images = []
    if len(images) >= 16:
        images = images[:16]  # Ensure we have exactly 16 images
        inputs = torch.stack(images).unsqueeze(0)  # Create batch dimension

        with torch.no_grad():
            output, feature = model(inputs)
            feature = F.normalize(feature, p=2, dim=1)
            out = classifier(feature)

        abnormal_percentage = (out.item() > abnormal_threshold) * 100
        overall_classification = 'Abnormal' if abnormal_percentage >= 50 else 'Normal'
    else:
        raise ValueError("Error: Not enough frames in the video")

    return extract_path, overall_classification, abnormal_percentage

# Define function to display custom-colored messages based on the input value
def display_alert(input_value, file_name):
    if input_value > 45:
        st.markdown('<div class="custom-alert">Abnormal activity detected!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="custom-alert-normal">No suspicious activity detected, Normal video</div>', unsafe_allow_html=True)
    
    # Display location if available
    name = os.path.splitext(file_name)[0]
    location = location_data.get(name, None)
    if location:
        st.write(f"Location: {location['City']}, {location['Street']}")
    else:
        st.write("Location data not available for this video.")


# Define function to display donut chart
def make_donut(input_response, input_text):
    if input_response > 45:
        chart_color = ['#E74C3C', '#781F16']  # Red color scheme
    else:
        chart_color = ['#27AE60', '#12783D']  # Green color scheme

    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100 - input_response, input_response]
    })

    plot = alt.Chart(source).mark_arc(innerRadius=60, outerRadius=70, cornerRadius=25).encode(
        theta="% value",
        color=alt.Color("Topic:N", scale=alt.Scale(domain=[input_text], range=chart_color), legend=None)
    ).properties(width=200, height=200)

    text = plot.mark_text(align='center', color=chart_color[0], font="Lato", fontSize=30, fontWeight=600,
                          fontStyle="italic").encode(text=alt.value(f'{input_response:.2f} %'))

    return plot + text

def display_donut_chart(input_value):
    if input_value is not None:
        st.altair_chart(make_donut(input_value, 'Abnormal Percentage'), use_container_width=True)

# Streamlit UI
uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "zip"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.zip'):
        with st.spinner('Processing ZIP file...'):
            try:
                extract_path, overall_classification, abnormal_percentage = process_zip(uploaded_file)
                st.success('Processing completed!')
                st.write(f"Overall Classification: {overall_classification}")
                st.write(f"Abnormal Percentage: {abnormal_percentage}%")
                display_donut_chart(abnormal_percentage)
                st.write("Extracted Images:")
                image_files = sorted(glob.glob(os.path.join(extract_path, '*.jpg')))
                for image_file in image_files:
                    st.image(image_file, caption=os.path.basename(image_file), use_column_width=True)
                display_alert(abnormal_percentage, uploaded_file.name)
            except ValueError as e:
                st.error(str(e))
    else:
        with st.spinner('Processing video...'):
            output_video_path, overall_classification, abnormal_percentage = process_video(uploaded_file)
            st.success('Processing completed!')
            st.write(f"Overall Classification: {overall_classification}")
            st.write(f"Abnormal Percentage: {abnormal_percentage}%")
            display_donut_chart(abnormal_percentage)
            st.video(output_video_path)
            display_alert(abnormal_percentage, uploaded_file.name)
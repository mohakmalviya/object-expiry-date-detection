import cv2
import os
import pandas as pd
import numpy as np
from ultralytics import YOLO
import streamlit as st

# Load the YOLO model
MODEL_PATH = '/home/senku/object-expiry-date-detection/best.pt'
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please ensure 'best.pt' is in the working directory.")
    st.stop()
model = YOLO(MODEL_PATH)

# Streamlit App Title
st.title("Object Detection App")
st.subheader("Live Object Detection with YOLOv8")

# Initialize output directory
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Initialize CSV file paths
csv_file_path = os.path.join(output_dir, 'item_counts.csv')
dates_csv_path = os.path.join(output_dir, 'count.csv')

# Load or create DataFrame
if os.path.exists(csv_file_path):
    data_table = pd.read_csv(csv_file_path)
else:
    data_table = pd.DataFrame(columns=['Frame Number', 'Item Count'])
data_table.to_csv(csv_file_path, index=False)

# Checkbox for using live camera
use_webcam = st.checkbox("Enable Live Camera Feed")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

frame_count = 0
max_item_count = 0

if use_webcam:
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not access the webcam.")
        st.stop()

    # Placeholder for video feed
    video_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame from webcam.")
            break

        frame_count += 1

        # Perform object detection
        results = model.predict(source=frame, conf=0.5, save=False)
        annotated_frame = results[0].plot()

        # Count detected items
        num_items = len(results[0].boxes)

        # Update maximum count
        global_max_updated = False
        if num_items > max_item_count:
            max_item_count = num_items
            max_frame = frame.copy()
            global_max_updated = True

        # Save annotated frame
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(
            annotated_frame_rgb,
            caption=f"Detected Items: {num_items} (Max: {max_item_count})",
            use_container_width=True,
        )

        # Update CSVs if maximum count changes
        if global_max_updated:
            image_path = os.path.join(output_dir, 'max_items_frame.png')
            cv2.imwrite(image_path, max_frame)

            # Append new row to DataFrame
            new_row = pd.DataFrame([{'Frame Number': frame_count, 'Item Count': max_item_count}])
            data_table = pd.concat([data_table, new_row], ignore_index=True)
            data_table.to_csv(csv_file_path, index=False)

            # Update maximum count CSV
            max_count_data = pd.DataFrame([{'Maximum Item Count': max_item_count}])
            max_count_data.to_csv(dates_csv_path, index=False)

    cap.release()
else:
    st.info("Enable the webcam to start live object detection.")

# Process uploaded image
if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    frame_count += 1

    # Perform object detection
    results = model.predict(source=image, conf=0.5, save=False)
    annotated_image = results[0].plot()

    # Count detected items
    num_items = len(results[0].boxes)

    # Update maximum count
    if num_items > max_item_count:
        max_item_count = num_items
        max_frame = image.copy()

        # Save annotated image
        image_path = os.path.join(output_dir, f'uploaded_image_{frame_count}.png')
        cv2.imwrite(image_path, max_frame)

        # Append new row to DataFrame
        new_row = pd.DataFrame([{'Frame Number': frame_count, 'Item Count': max_item_count}])
        data_table = pd.concat([data_table, new_row], ignore_index=True)
        data_table.to_csv(csv_file_path, index=False)

        # Update maximum count CSV
        max_count_data = pd.DataFrame([{'Maximum Item Count': max_item_count}])
        max_count_data.to_csv(dates_csv_path, index=False)

    # Convert image to RGB for display
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    st.image(annotated_image_rgb, caption=f"Detected Items: {num_items} (Max: {max_item_count})", use_container_width=True)
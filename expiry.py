import streamlit as st
import cv2
import easyocr
import re
import os
import pandas as pd
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# Load YOLO model
MODEL_PATH = 'best_model.pt'
model = YOLO(MODEL_PATH)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Create output folder
if not os.path.exists('output'):
    os.makedirs('output')

# Initialize session state for expiry_table
if 'expiry_table' not in st.session_state:
    if os.path.exists('output/expiry_dates.csv'):
        st.session_state.expiry_table = pd.read_csv('output/expiry_dates.csv')
    else:
        st.session_state.expiry_table = pd.DataFrame(columns=['Image ID', 'Expiry Date', 'Days Remaining'])

# Define regex pattern for dates
date_pattern = re.compile(r"(\d{2}[/-]\d{2}[/-]\d{4})")

def calculate_days_remaining(expiry_date_str):
    try:
        expiry_date = datetime.strptime(expiry_date_str, '%d-%m-%Y')
    except ValueError:
        try:
            expiry_date = datetime.strptime(expiry_date_str, '%d/%m/%Y')
        except ValueError:
            return "Invalid Date"
    current_date = datetime.now()
    days_remaining = (expiry_date - current_date).days
    if days_remaining < 0:
        return "Expired"
    return days_remaining

def preprocess_frame(frame):
    """Preprocess frame for better OCR accuracy"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

def process_uploaded_image(image, image_id):
    days_remaining = "No data"  # Initialize days_remaining to a default value

    # Perform object detection with YOLO
    results = model.predict(source=image, conf=0.5, save=False)
    annotated_frame = results[0].plot()

    # Process frame for OCR
    processed_frame = preprocess_frame(image)
    result = reader.readtext(processed_frame)

    display_frame = annotated_frame.copy()
    detected_expiry_date = None

    # Store detected texts with positions
    detected_texts = []
    for (bbox, text, prob) in result:
        if prob > 0.6:
            x, y = int(bbox[0][0]), int(bbox[0][1])
            detected_texts.append({'text': text.strip(), 'x': x, 'y': y})

    # Look for expiry keyword and associated date
    for idx, item in enumerate(detected_texts):
        text = item['text']
        if re.search(r'\b(expiry|exp|expires)\b', text, re.IGNORECASE):
            # Check next text items for date
            for next_item in detected_texts[idx+1:]:
                date_match = re.search(date_pattern, next_item['text'])
                if date_match:
                    expiry_date_str = date_match.group(1)
                    detected_expiry_date = expiry_date_str
                    print(f"Extracted expiry date: {detected_expiry_date}")  # Debugging
                    break
            if detected_expiry_date:
                break  # Exit if expiry date found

    if detected_expiry_date:
        cv2.putText(display_frame, f"Expiry Date: {detected_expiry_date}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        days_remaining = calculate_days_remaining(detected_expiry_date)

        # Save to session state DataFrame and CSV
        if st.session_state.expiry_table.empty or detected_expiry_date not in st.session_state.expiry_table['Expiry Date'].values:
            new_row = pd.DataFrame([{
                'Image ID': image_id,
                'Expiry Date': detected_expiry_date,
                'Days Remaining': days_remaining
            }])
            st.session_state.expiry_table = pd.concat([st.session_state.expiry_table, new_row], ignore_index=True)
            st.session_state.expiry_table.to_csv('output/expiry_dates.csv', index=False)

            print(f"Saved expiry date: {detected_expiry_date}")  # Debugging

    return display_frame, detected_expiry_date, days_remaining

st.title("Expiry Date Detection")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_id = uploaded_file.name
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    display_frame, detected_expiry_date, days_remaining = process_uploaded_image(image, image_id)

    # Convert BGR to RGB for displaying with Streamlit
    display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    st.image(display_frame_rgb, channels="RGB", use_container_width=True)

    if detected_expiry_date:
        st.write(f"**Expiry Date:** {detected_expiry_date}")
        st.write(f"**Days Remaining:** {days_remaining}")
        st.write("Expiry date saved to CSV.")
    else:
        st.write("No expiry date detected.")

# Display the contents of the CSV file
if not st.session_state.expiry_table.empty:
    st.write("### Detected Expiry Dates")
    st.dataframe(st.session_state.expiry_table)
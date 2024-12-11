import streamlit as st
import cv2
import pytesseract
import re
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize session state
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# Create output folder
if not os.path.exists('output'):
    os.makedirs('output')

# Initialize DataFrame
columns = ['Frame Number', 'Expiry Date', 'Days Remaining']
expiry_table = pd.DataFrame(columns=columns)

# Define patterns
expiry_pattern = re.compile(r"(expiry|EXP|exp|expires)?[:\s]*(\d{2}[/-]\d{2}[/-]\d{2,4}|\d{2}[/-]\d{4}|\d{2}/\d{4})", re.IGNORECASE)

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
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)
    return gray

def extract_expiry_date(text):
    """Extract expiry date using regex"""
    match = re.search(expiry_pattern, text)
    if match:
        return match.group(2)
    return None

def start_camera():
    st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.camera_running = True

def stop_camera():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
    st.session_state.camera_running = False

st.title("Expiry Date Detection")

if st.button("Toggle Camera"):
    if st.session_state.camera_running:
        stop_camera()
    else:
        start_camera()

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Create placeholder for video feed
frame_placeholder = st.empty()
info_placeholder = st.empty()

# Main detection loop for camera feed
if st.session_state.camera_running and st.session_state.cap is not None:
    frame_count = 0
    detected_dates = set()
    
    while True:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to read from camera")
            break

        frame_count += 1

        processed_frame = preprocess_frame(frame)

        data = pytesseract.image_to_data(processed_frame, config='--psm 6', output_type=pytesseract.Output.DICT)

        display_frame = frame.copy()
        detected_expiry_date = None

        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 90:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                text = data['text'][i].strip()

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                expiry_date = extract_expiry_date(text)
                if expiry_date and expiry_date not in detected_dates:
                    detected_expiry_date = expiry_date
                    detected_dates.add(expiry_date)

        if detected_expiry_date:
            cv2.putText(display_frame, f"Expiry Date: {detected_expiry_date}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            days_remaining = calculate_days_remaining(detected_expiry_date)

            image_path = f'output/frame_{frame_count}.png'
            cv2.imwrite(image_path, frame)

            if detected_expiry_date not in expiry_table['Expiry Date'].values:
                new_row = pd.DataFrame([{'Frame Number': frame_count, 'Expiry Date': detected_expiry_date, 'Days Remaining': days_remaining}])
                expiry_table = pd.concat([expiry_table, new_row], ignore_index=True)

                expiry_table.to_csv('output/expiry_dates.csv', index=False)

        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        info_text = ""
        if detected_expiry_date:
            info_text += f"Expiry Date: {detected_expiry_date}\n"
            info_text += f"Days Remaining: {days_remaining}\n"
        
        if info_text:
            info_placeholder.text(info_text)

        # Break if stop button pressed
        if not st.session_state.camera_running:
            break

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    processed_image = preprocess_frame(image)

    data = pytesseract.image_to_data(processed_image, config='--psm 6', output_type=pytesseract.Output.DICT)

    display_image = image.copy()
    detected_expiry_date = None

    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 90:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            text = data['text'][i].strip()

            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            expiry_date = extract_expiry_date(text)
            if expiry_date and expiry_date not in detected_dates:
                detected_expiry_date = expiry_date
                detected_dates.add(expiry_date)

    if detected_expiry_date:
        cv2.putText(display_image, f"Expiry Date: {detected_expiry_date}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        days_remaining = calculate_days_remaining(detected_expiry_date)

        image_path = f'output/uploaded_image.png'
        cv2.imwrite(image_path, image)

        if detected_expiry_date not in expiry_table['Expiry Date'].values:
            new_row = pd.DataFrame([{'Frame Number': 'Uploaded', 'Expiry Date': detected_expiry_date, 'Days Remaining': days_remaining}])
            expiry_table = pd.concat([expiry_table, new_row], ignore_index=True)

            expiry_table.to_csv('output/expiry_dates.csv', index=False)

    image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(image_rgb, channels="RGB", use_container_width=True)

    info_text = ""
    if detected_expiry_date:
        info_text += f"Expiry Date: {detected_expiry_date}\n"
        info_text += f"Days Remaining: {days_remaining}\n"
    
    if info_text:
        info_placeholder.text(info_text)

# Cleanup
if not st.session_state.camera_running and st.session_state.cap is not None:
    st.session_state.cap.release()
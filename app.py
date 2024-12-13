from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import pytesseract
import re
from datetime import datetime
from ultralytics import YOLO
import base64
import pandas as pd
import os

app = Flask(__name__)

# Load YOLO model for item detection
model = YOLO('best.pt')

# Regex for expiry date detection

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
item_count_csv = os.path.join(output_dir, 'item_counts.csv')
expiry_csv = os.path.join(output_dir, 'expiry_dates.csv')

# Load or create DataFrame for item counts
if os.path.exists(item_count_csv):
    item_table = pd.read_csv(item_count_csv)
else:
    item_table = pd.DataFrame(columns=['Frame Number', 'Item Count'])
    item_table.to_csv(item_count_csv, index=False)

# Load or create DataFrame for expiry dates
if os.path.exists(expiry_csv):
    expiry_table = pd.read_csv(expiry_csv)
else:
    expiry_table = pd.DataFrame(columns=['Frame Number', 'Expiry Date', 'Days Remaining'])
    expiry_table.to_csv(expiry_csv, index=False)

max_item_count = 0 

def preprocess_frame(frame):
    """Preprocess frame for better OCR accuracy."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)
    return gray

def extract_expiry_date(text):
    """Extract expiry date using regex."""
    expiry_pattern = re.compile(r"(expiry|EXP|exp|expires)?[:\s]*(\d{2}[/-]\d{2}[/-]\d{2,4}|\d{2}[/-]\d{4}|\d{2}/\d{4})", re.IGNORECASE)
    match = re.search(expiry_pattern, text)
    if match:
        return match.group(2)
    return None

def calculate_days_remaining(expiry_date_str):
    """Calculate days remaining until expiry."""
    try:
        expiry_date = datetime.strptime(expiry_date_str, '%d-%m-%Y')
    except ValueError:
        try:
            expiry_date = datetime.strptime(expiry_date_str, '%d/%m/%Y')
        except ValueError:
            return "Invalid Date"
    current_date = datetime.now()
    days_remaining = (expiry_date - current_date).days
    return "Expired" if days_remaining < 0 else days_remaining

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/expiry_detection')
def expiry_detection():
    return render_template('expiry_date.html')

@app.route('/process_item_count', methods=['POST'])
def process_item_count():
    """Process frames for item counting."""
    global max_item_count, item_table

    data = request.json.get('image')
    if not data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        img_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Invalid image data: {str(e)}'}), 400

    # Perform object detection with YOLO
    results = model.predict(source=frame, conf=0.5, save=False)
    annotated_frame = results[0].plot()  # Annotate frame with predictions
    num_items_detected = len(results[0].boxes)  # Count detected items

    # Update maximum count if needed
    if num_items_detected > max_item_count:
        max_item_count = num_items_detected

        # Save to DataFrame and CSV
        new_row = pd.DataFrame([{'Frame Number': len(item_table) + 1, 'Item Count': max_item_count}])
        item_table = pd.concat([item_table, new_row], ignore_index=True)
        item_table.to_csv(item_count_csv, index=False)

    # Encode annotated frame to base64 string
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    annotated_frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'item_count': num_items_detected, 'annotated_frame': annotated_frame_base64})

@app.route('/max_item_count')
def get_max_item_count():
    """Return the maximum item count as JSON."""
    global max_item_count
    return jsonify(max_item_count=max_item_count)

@app.route('/process_expiry_date', methods=['POST'])
def process_expiry_date():
    """Process frames for expiry date detection."""
    global expiry_table
    
    data = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Preprocess frame for OCR
    processed_frame = preprocess_frame(frame)
    
    # Perform OCR
    data = pytesseract.image_to_data(processed_frame, config='--psm 6', output_type=pytesseract.Output.DICT)

    detected_expiry_date = None
    days_remaining = "No data"

    # Process OCR results and draw on frame
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60:  # Adjust confidence threshold as needed
            x, y, w, h = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            text = data['text'][i].strip()

            # Draw bounding box and text on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Check for expiry date in detected text
            expiry_date = extract_expiry_date(text)
            if expiry_date:
                detected_expiry_date = expiry_date
                days_remaining = calculate_days_remaining(detected_expiry_date)

                # Save to DataFrame if new date is found and save to CSV
                if detected_expiry_date not in expiry_table['Expiry Date'].values:
                    new_row = pd.DataFrame([{
                        'Frame Number': len(expiry_table) + 1,
                        'Expiry Date': detected_expiry_date,
                        'Days Remaining': days_remaining
                    }])
                    expiry_table = pd.concat([expiry_table, new_row], ignore_index=True)
                    expiry_table.to_csv(expiry_csv, index=False)

                break

    # Retrieve last known expiry date if none detected in current frame
    if not detected_expiry_date and not expiry_table.empty:
        latest_entry = expiry_table.iloc[-1]
        detected_expiry_date = latest_entry['Expiry Date']
        days_remaining = latest_entry['Days Remaining']

    # Annotate frame with last known or current expiry date information
    cv2.putText(frame, f"Expiry Date: {detected_expiry_date}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)
    cv2.putText(frame, f"Days Remaining: {days_remaining}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

    # Encode annotated frame to base64 string
    _, buffer = cv2.imencode('.jpg', frame)
    annotated_frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'expiry_date': detected_expiry_date or "No data",
        'days_remaining': days_remaining,
        'annotated_frame': annotated_frame_base64
    })

if __name__ == "__main__":
    app.run(debug=True)

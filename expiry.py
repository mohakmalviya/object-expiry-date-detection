import cv2
import pytesseract
import re
import os
import pandas as pd
from datetime import datetime

if not os.path.exists('output'):
    os.makedirs('output')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

tess_config = '--psm 6'

expiry_pattern = re.compile(r"(expiry|exp|EXP|expires)?[:\s]*(\d{2}[/-]\d{2}[/-]\d{2,4}|\d{2}[/-]\d{4}|\d{2}/\d{4})", re.IGNORECASE)

columns = ['Frame Number', 'Expiry Date', 'Days Remaining']
expiry_table = pd.DataFrame(columns=columns)

detected_dates = set()

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

def calculate_days_remaining(expiry_date_str):
    """Calculate the number of days remaining until the expiry date"""
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

def main():
    global expiry_table
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        frame_count += 1
        processed_frame = preprocess_frame(frame)

        # Perform OCR with bounding boxes
        data = pytesseract.image_to_data(processed_frame, config=tess_config, output_type=pytesseract.Output.DICT)

        print("\nFull OCR Output:")
        print(data['text'])

        display_frame = frame.copy()
        detected_expiry_date = None

        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 90:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                text = data['text'][i].strip()

                print(f"Detected text: '{text}', Confidence: {data['conf'][i]}")

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                expiry_date = extract_expiry_date(text)
                if expiry_date and expiry_date not in detected_dates:
                    detected_expiry_date = expiry_date
                    detected_dates.add(expiry_date)

        if detected_expiry_date:
            cv2.putText(display_frame, f"Expiry Date: {detected_expiry_date}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(f"Detected Expiry Date: {detected_expiry_date}")

            days_remaining = calculate_days_remaining(detected_expiry_date)
            print(f"Days Remaining: {days_remaining}")

            image_path = f'output/frame_{frame_count}.png'
            cv2.imwrite(image_path, frame)
            print(f"Captured image saved to {image_path}")

            # Check if the expiry date is already in the DataFrame
            if detected_expiry_date not in expiry_table['Expiry Date'].values:
                new_row = pd.DataFrame([{'Frame Number': frame_count, 'Expiry Date': detected_expiry_date, 'Days Remaining': days_remaining}])
                expiry_table = pd.concat([expiry_table, new_row], ignore_index=True)
                print(f"New row added: {new_row}")

                expiry_table.to_csv('output/expiry_dates.csv', index=False)
                print("\nExpiry dates saved to 'output/expiry_dates.csv'")

        cv2.imshow('OCR with Bounding Boxes', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
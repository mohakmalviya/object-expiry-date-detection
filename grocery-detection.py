import cv2
from ultralytics import YOLO
import os
import pandas as pd

# Load the YOLO model
model = YOLO('/home/senku/object-expiry-date-detection/best.pt')

# Initialize the webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create output folder if not exists
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Initialize the CSV file paths
csv_file_path = os.path.join(output_dir, 'item_counts.csv')
dates_csv_path = os.path.join(output_dir, 'count.csv')

# Initialize DataFrame
try:
    if os.path.exists(csv_file_path):
        data_table = pd.read_csv(csv_file_path)
        print(f"Loaded existing CSV file: {csv_file_path}")
    else:
        columns = ['Frame Number', 'Item Count']
        data_table = pd.DataFrame(columns=columns)
        data_table.to_csv(csv_file_path, index=False)
        print(f"Created new CSV file: {csv_file_path}")
except Exception as e:
    print(f"Error with CSV file operations: {e}")
    exit()

frame_count = 0
max_item_count = 0
max_frame = None

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    frame_count += 1

    # Perform object detection
    results = model.predict(source=frame, conf=0.5, save=False)
    annotated_frame = results[0].plot()

    # Count the number of detected items
    num_items = len(results[0].boxes)
    print(f"Number of items detected: {num_items}")

    # Update maximum count if needed
    if num_items > max_item_count:
        max_item_count = num_items
        max_frame = frame.copy()
        print(f"New maximum count detected: {max_item_count}")

        try:
            # Save the new maximum frame
            image_path = os.path.join(output_dir, 'max_items_frame.png')
            cv2.imwrite(image_path, max_frame)
            print(f"Saved new maximum frame to: {image_path}")

            # Update CSV with new maximum
            new_row = pd.DataFrame([{'Frame Number': frame_count, 'Item Count': max_item_count}])
            data_table = pd.concat([data_table, new_row], ignore_index=True)
            data_table.to_csv(csv_file_path, index=False)
            print(f"Updated item counts in: {csv_file_path}")

            # Update dates.csv with new maximum
            max_count_data = pd.DataFrame([{'Maximum Item Count': max_item_count}])
            max_count_data.to_csv(dates_csv_path, index=False)
            print(f"Updated maximum count in: {dates_csv_path}")
        except Exception as e:
            print(f"Error saving data: {e}")

    # Display the annotated frame
    cv2.putText(annotated_frame, f"Items: {num_items} (Max: {max_item_count})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Camera Feed", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()

print(f"Final maximum item count: {max_item_count}")
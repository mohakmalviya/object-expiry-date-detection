import cv2
from ultralytics import YOLO

model = YOLO('/home/senku/vision/best.pt')

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    results = model.predict(source=frame, conf=0.5, save=False)

    annotated_frame = results[0].plot()

    cv2.imshow("Live Camera Feed", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

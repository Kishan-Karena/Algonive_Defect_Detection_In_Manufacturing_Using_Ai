import cv2
from ultralytics import YOLO

# 1. Load your custom trained model
# Make sure the path to your best.pt is correct
model = YOLO("runs/detect/train/weights/best.pt")

# 2. Use 0 for the default system webcam
# If you have an external USB camera, you might need to try 1 or 2
cap = cv2.VideoCapture(0)

# Check if the camera opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Optional: Set resolution (for better performance or higher detail)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting real-time detection... Press 'q' to exit.")

# 3. Loop through the camera frames
while True:
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference
        # stream=True is more memory-efficient for real-time video
        results = model(frame, conf=0.5, stream=True)

        # Iterate through the generator results
        for r in results:
            # Visualize the detections on the frame
            annotated_frame = r.plot()

            # Display the annotated frame
            cv2.imshow("Real-Time Defect Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Failed to grab frame from camera.")
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

all_class_names = ['Defective', 'cast', 'hole', 'scratches']

video_path = "Video3.mp4" 
cap = cv2.VideoCapture(video_path)

display_classes = ['Defective', 'hole', 'scratches']
total_counts = {name: 0 for name in display_classes}
already_counted_ids = set()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True, conf=0.3)

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.int().cpu().tolist()

        for track_id, cls_index in zip(track_ids, clss):
            label = all_class_names[cls_index]
            
            if label != 'cast' and track_id not in already_counted_ids:
                total_counts[label] += 1
                already_counted_ids.add(track_id)

    annotated_frame = results[0].plot()

    cv2.rectangle(annotated_frame, (5, 5), (250, 120), (0, 0, 0), -1) 
    cv2.putText(annotated_frame, "DEFECT REPORT:", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_offset = 55
    for name, count in total_counts.items():
        cv2.putText(annotated_frame, f"{name}: {count}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30

    cv2.imshow("Defect Tracking (Excluding Cast)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
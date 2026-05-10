import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model.predict("Cast3.jpeg", save=False)

res = results[0]

annotated_img = res.plot()

cv2.imshow("YOLOv8 Detection", annotated_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
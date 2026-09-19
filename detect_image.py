from ultralytics import YOLO

# --------------------------------
# Load Pre-trained YOLO Model
# --------------------------------

model = YOLO("yolo11n.pt")

print("YOLO Model Loaded Successfully!")

# --------------------------------
# Image Path
# --------------------------------

image_path = "images/image/img2.jpeg"

# --------------------------------
# Run Object Detection
# --------------------------------

results = model.predict(
    source=image_path,
    conf=0.25,
    save=True
)

# --------------------------------
# Display Results
# --------------------------------

for result in results:

    print("\nDetected Objects:")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]

        print(
            f"{class_name} : "
            f"{confidence * 100:.2f}%"
        )

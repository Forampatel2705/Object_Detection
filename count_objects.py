from ultralytics import YOLO
from collections import Counter

# --------------------------------
# Load Pre-trained YOLO Model
# --------------------------------

model = YOLO("yolo11n.pt")

print("YOLO Model Loaded Successfully!")

# --------------------------------
# Run Detection
# --------------------------------

results = model.predict(
    source="images/img2.jpg",
    conf=0.25
)

# --------------------------------
# Count Detected Objects
# --------------------------------

counter = Counter()

for result in results:

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        counter[class_name] += 1

# --------------------------------
# Display Counts
# --------------------------------

print("\n===== OBJECT COUNT =====")

for object_name, count in counter.items():
    print(
        f"{object_name} : {count}"
    )

from ultralytics import YOLO

# --------------------------------
# Load Pre-trained YOLO Model
# --------------------------------

model = YOLO("yolo11n.pt")

print("Starting Webcam...")
print("Press Q to exit.")

# --------------------------------
# Real-Time Webcam Detection
# source=0 → default webcam
# show=True → display live window
# --------------------------------

results = model.predict(
    source=0,
    show=True,
    conf=0.40
)
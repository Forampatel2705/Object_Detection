from ultralytics import YOLO

# --------------------------------
# Load Pre-trained YOLO Model
# --------------------------------

model = YOLO("yolo11n.pt")

print("Starting Object Tracking...")
print("Press Q to exit.")

# --------------------------------
# Real-Time Object Tracking
# source=0    → default webcam
# show=True   → display live window
# tracker     → tracking algorithm
# --------------------------------

model.track(
    source=0,
    show=True,
    tracker="bytetrack.yaml"
)

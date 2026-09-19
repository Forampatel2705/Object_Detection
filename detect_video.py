from ultralytics import YOLO

# --------------------------------
# Load Pre-trained YOLO Model
# --------------------------------

model = YOLO("yolo11n.pt")

print("YOLO Model Loaded Successfully!")

# --------------------------------
# Video Path
# --------------------------------

video_path = "videos/videos.mp4"

# --------------------------------
# Detect Objects (frame by frame)
# --------------------------------

results = model.predict(
    source=video_path,
    conf=0.25,
    save=True
)

print("Video Detection Completed!")

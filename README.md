 Follow this order:

Step 1 — Environment
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

Step 2 — Install
python -m pip install --upgrade pip
pip install -r requirements.txt

Step 3 — Verify
python -c "import ultralytics; print('Ultralytics OK')"
python -c "import cv2; print('OpenCV OK')"

Step 4 — Image
python detect_image.py

Step 5 — Video
python detect_video.py

Step 6 — Webcam
python webcam.py

Step 7 — Image Counting
python count_objects.py

Step 8 — Real-Time Counting
python webcam_count.py





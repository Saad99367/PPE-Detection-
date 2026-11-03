# PPE Detection System

Python, Streamlit, OpenCV, and a local custom YOLO11 checkpoint for image,
browser webcam, and uploaded-video PPE detection.

## Run

Use Python 3.11:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Place your trained checkpoint at `models/best.pt`. No replacement model is
downloaded. Class IDs are read from the checkpoint, never assumed.

```powershell
.\.venv\Scripts\python.exe predict.py --source images/test1.jpg
.\.venv\Scripts\python.exe webcam.py --source 0 --show
.\.venv\Scripts\python.exe webcam.py --source videos/video1.mp4 --show
```

For local GUI playback, replace headless OpenCV in your environment:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y opencv-python-headless opencv-python
.\.venv\Scripts\python.exe -m pip install opencv-python
```

## Detection appearance

`visualization.py` supplies one OpenCV renderer to `predict.Detector.predict`.
The dashboard, browser callback, upload worker, and both CLIs all use it.
Defaults at 1280px width are 5px anti-aliased borders and 1.0 font scale,
with padded solid labels and contrast-selected text. Sizes scale with frame
width, with minimum sizes for smaller frames. Images retain their dimensions
and aspect ratio. Extremely tiny frames cannot accommodate fully legible labels.

- Green: Hardhat, Mask, Safety Vest.
- Red: NO-Hardhat, NO-Mask, NO-Safety Vest.
- Blue: additional classes.

Labels include the class name and rounded confidence percentage. Label placement
prefers above the box, then inside/below, choosing the least overlapping candidate.
Crowded scenes can still have overlapping labels. Coordinates and labels are
clipped/fitted to the image; original detection data remains unchanged.

The **Detection Appearance** sidebar controls reference box thickness (2–8),
font size, confidence visibility, and class label visibility. For images, press
**Detect PPE** again after changing controls. Live webcam and active video jobs
apply appearance changes to subsequent frames. Inference thresholds for uploaded
video are captured at Start. Appearance updates do not reload the cached model.
Immutable settings are exchanged under locks between UI and worker threads.

## Privacy and operation

No annotated images or videos are saved. Uploaded video input is temporarily
written for OpenCV decoding and removed on completion, stop, or worker failure.
Inactive sessions time out after 30 seconds, subject to an in-flight inference
finishing. A force-killed process may leave a temporary input behind.
Browser frames are sent to the app server for inference. Browser camera access
requires permission and HTTPS (localhost is exempt); remote networks may require
a TURN server. The default uses Google's public STUN service.

CUDA is selected when available, otherwise CPU. Smaller inference sizes can
improve throughput. Model inference is serialized for safe multi-session use.
No accuracy or throughput benchmarks are claimed. This is decision support;
missed or incorrect detections are possible and human supervision is required.

## Validation

Rendering smoke checks cover small, HD, and 1080p frames, light/dark backgrounds,
out-of-bounds boxes, empty detections, label toggles, and unchanged input pixels
and detection coordinates. Interactive webcam/video operation requires validation
in your installed application environment.

`models/best.pt` is eligible for Git tracking. Use Git LFS for large checkpoints
that exceed your hosting provider's regular Git file limit. Do not commit private
media or `.streamlit/secrets.toml`.

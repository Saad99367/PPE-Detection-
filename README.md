# PPE Detection System

Detect personal protective equipment using a custom-trained YOLO11 model, Python, OpenCV, and Streamlit.

The **Streamlit dashboard accepts image uploads only**. Use the separate **`webcam.py` command-line tool** for a local camera or video file.

## Features

- Upload JPG, JPEG, or PNG images and view annotated results.
- View class counts, individual confidence scores, bounding-box coordinates, and inference time.
- Adjust confidence, IoU, inference image size, and annotation appearance.
- Run local camera or video inference with an optional OpenCV preview.
- Fit video previews within configurable dimensions while preserving aspect ratio.
- Automatically use CUDA when available, otherwise CPU.
- Reuse a cached, synchronized model in the dashboard.
- Process predictions in memory without saving annotated images or videos.

## Supported classes

| Category | Classes | Box color |
| --- | --- | --- |
| PPE present | Hardhat, Mask, Safety Vest | Green |
| PPE violation | NO-Hardhat, NO-Mask, NO-Safety Vest | Red |
| Additional model classes | Names supplied by the checkpoint | Blue |

Class IDs and names come from `model.names`; their numeric order is never assumed.

## Project structure

```text
PPE-Detection/
├── app.py                   # Image-only Streamlit dashboard
├── predict.py               # Model loading, image inference, and image CLI
├── webcam.py                # Local camera/video CLI and reusable processing helpers
├── visualization.py         # Shared OpenCV boxes and labels
├── models/
│   └── best.pt              # Your trained checkpoint
├── images/                  # Optional local test images (Git-ignored)
├── videos/                  # Optional local test videos (Git-ignored)
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml          # Dark theme and upload limit
```

All inference entry points call `Detector.predict()` in `predict.py`, which uses the renderer in `visualization.py`. The dashboard does not expose browser camera or video upload modes. Reusable browser/upload helpers remain in `webcam.py` but are not connected to `app.py`.

## Installation

Python **3.11** is recommended. Open a terminal in the project directory.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

These commands use the environment directly, so PowerShell activation is unnecessary. If `python` is not found, use `py -3.11` when creating the environment or the full path to your Python installation.

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the commands below, `python` must refer to the environment where dependencies were installed. In PowerShell you can replace it with `.\.venv\Scripts\python.exe`.

### Model setup

Place your custom checkpoint at `models/best.pt`. The default path is resolved relative to the project code, not the terminal's working directory. A missing checkpoint produces an error; the application does not train or download a replacement model.

Use only checkpoints you trust. Keep `best.pt` eligible for Git tracking; use Git LFS if the checkpoint is too large for regular Git hosting. Share the model only if you have the rights to do so.

## Image dashboard

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

1. Upload a JPG, JPEG, or PNG image.
2. Choose detection and appearance settings in the sidebar.
3. Click **Detect PPE**.
4. Review the annotated image, counts, confidence scores, and inference time.

After changing settings, click **Detect PPE** again to update the image. The configured upload limit is 200 MB. No secrets file or camera permissions are needed for this dashboard. Stop the server with **Ctrl+C**.

## Image command line

```bash
python predict.py --source images/test1.jpg
python predict.py --source images/test1.jpg --conf 0.4 --iou 0.45 --imgsz 640
python predict.py --source images/test1.jpg --model models/best.pt --show
```

The CLI prints JSON containing detections, class counts, and inference time. `--show` opens an OpenCV window; press any key in that window to close it.

## Local webcam and video

### Enable OpenCV windows

The dependency file includes headless OpenCV for server use. For local `--show` playback, install the GUI build instead. Avoid keeping both OpenCV distributions installed in the same environment.

```bash
python -m pip uninstall -y opencv-python-headless opencv-python
python -m pip install "opencv-python>=4.10,<5"
```

### Webcam

```bash
python webcam.py --source 0 --show
```

`0` selects the default camera. Try `--source 1` for another connected camera. Windows must allow desktop applications to access the camera.

### Video file

```bash
python webcam.py --source videos/video1.mp4 --show
python webcam.py --source videos/video1.mp4 --show --display-width 960 --display-height 540
```

Video previews fit inside **1280 × 720** by default. Resizing preserves aspect ratio and does not upscale small videos. It affects only the displayed annotated video, not model inference resolution or webcam frames.

Press **Q** or **Esc** in the preview window to stop. Without `--show`, frames are processed and approximate processing FPS is printed in the terminal; use **Ctrl+C** to stop. Video processing runs at the available processing speed, not necessarily the source playback rate.

### CLI options

| Option | Default | Purpose |
| --- | --- | --- |
| `--source` | Required for image CLI; `0` for webcam CLI | Image path, or camera index/video path respectively |
| `--model` | Project `models/best.pt` | Local checkpoint |
| `--conf` | `0.25` | Confidence threshold, 0–1 |
| `--iou` | `0.45` | IoU threshold, 0–1 |
| `--imgsz` | `640` | Inference size; positive multiple of 32 |
| `--show` | Off | Display an OpenCV preview |
| `--display-width` | `1280` | Video preview maximum width; webcam CLI only |
| `--display-height` | `720` | Video preview maximum height; webcam CLI only |

## Annotation appearance

The shared renderer uses anti-aliased boxes, padded solid label backgrounds, and contrasting text. At 1280px frame width, defaults are approximately **5px borders** and **1.0 font scale**. Sizes scale with frame width, with minimum sizes for smaller images.

Labels show the class name and rounded confidence percentage, such as `Hardhat 91%`. Placement prefers above the box, then alternative positions to reduce overlap. Coordinates are clipped for drawing without modifying prediction data. Dense scenes can still have overlapping labels; very small images limit text readability.

The dashboard's **Detection Appearance** section offers:

- Bounding Box Thickness: 2–8 at the reference width.
- Label Font Size.
- Show Confidence.
- Show Class Labels.

Appearance changes do not reload the model. CLI previews use the shared default appearance; dashboard controls do not configure separate CLI processes.

## Privacy and resource handling

The dashboard processes uploaded images on the machine running Streamlit. On remote hosting, images are transmitted to that server. Local CLI sources are processed on the computer running the command. Neither path saves annotated media, and model prediction saving is disabled.

The image dashboard does not create temporary video files. The retained, unused `VideoJob` helper writes an uploaded input to a temporary file if invoked separately and removes it when processing ends or stops. A force-killed process can leave such an input behind. Camera and video handles are released when CLI processing exits.

## Performance and deployment

CUDA is selected when PyTorch reports an available CUDA device; otherwise inference uses CPU. GPU execution requires a compatible PyTorch/CUDA installation. Smaller `--imgsz` values can reduce processing cost, with a possible accuracy tradeoff. Cached model access is serialized for thread safety, so concurrent sessions share inference capacity.

To host the image dashboard, install the requirements, provision the checkpoint, and launch:

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Use your hosting platform's access controls and HTTPS configuration for a remote deployment. Provision enough memory for PyTorch and the model. GUI CLI playback requires a local desktop and is not intended for a headless server. `streamlit-webrtc` and `av` remain listed for the retained helpers; the image dashboard does not import them directly.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `python` is not recognized | Use `py -3.11`, an installed Python path, or the virtual environment executable |
| Missing module | Install requirements using the same interpreter used to launch the app |
| Model not found | Place your trained checkpoint at `models/best.pt` or pass `--model` to the CLI |
| OpenCV GUI error | Replace headless OpenCV with the GUI package as described above |
| Camera cannot open | Check OS permissions, camera index, and whether another application is using it |
| Video cannot open | Check the path and decoder support; try an H.264 MP4 file |
| No detections | Check confidence threshold, image quality, and whether the checkpoint matches the intended classes |
| Slow inference | Reduce image size and check whether CUDA is available |
| Appearance changes are not visible | Click Detect PPE again; restart the running app after code updates if needed |

## Validation and limitations

Completed checks include module syntax/import checks, renderer smoke checks across small/HD/1080p dimensions and light/dark backgrounds, immutable input pixels and coordinates, empty detections, label visibility, and landscape/portrait/small-frame video resizing. Full interactive camera and video playback have not been validated in every target environment. No accuracy or FPS benchmarks are claimed.

This system is a visual decision-support tool, not a safety guarantee. Occlusion, lighting, distance, and unfamiliar scenes can cause missed or incorrect detections. Human safety supervision remains necessary.

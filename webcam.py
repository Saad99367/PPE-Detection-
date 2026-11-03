"""Browser frame callbacks, bounded upload workers, and local video CLI."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic, perf_counter

import cv2
import numpy as np

from visualization import Appearance

from predict import Detector, Prediction, Settings, add_inference_arguments, load_model


class BrowserProcessor:
    """A callback with thread-safe settings/status and no Streamlit calls."""

    def __init__(self, detector: Detector, settings: Settings, appearance: Appearance = Appearance()) -> None:
        self.detector = detector
        self.lock = Lock()
        self.settings = settings
        self.appearance = appearance
        self.error: str | None = None

    def configure(self, settings: Settings, appearance: Appearance = Appearance()) -> None:
        with self.lock:
            self.settings = settings
            self.appearance = appearance

    def status(self) -> str | None:
        with self.lock:
            return self.error

    def __call__(self, frame):
        import av
        with self.lock:
            settings = self.settings
            appearance = self.appearance
        try:
            result = self.detector.predict(frame.to_ndarray(format="bgr24"), settings, appearance)
            output = av.VideoFrame.from_ndarray(result.image_bgr, format="bgr24")
            output.pts, output.time_base = frame.pts, frame.time_base
            with self.lock:
                self.error = None
            return output
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
            return frame


class VideoJob:
    """Keep only the newest frame; delete the input on EOF, stop, or idle timeout."""

    def __init__(self, data: bytes, suffix: str, detector: Detector, settings: Settings, appearance: Appearance = Appearance()) -> None:
        self.lock = Lock()
        self.stop_event = Event()
        self.done = Event()
        self.appearance = appearance
        self.last_poll = monotonic()
        self.result: Prediction | None = None
        self.frames = 0
        self.fps = 0.0
        self.error: str | None = None
        self.path: Path | None = None
        self.thread = Thread(target=self._run, args=(data, suffix, detector, settings), daemon=True)
        self.thread.start()

    def configure_appearance(self, appearance: Appearance) -> None:
        with self.lock:
            self.appearance = appearance

    def snapshot(self):
        with self.lock:
            self.last_poll = monotonic()
            return self.result, self.frames, self.fps, self.error, self.done.is_set()

    def stop(self) -> None:
        self.stop_event.set()

    def _run(self, data: bytes, suffix: str, detector: Detector, settings: Settings, appearance: Appearance = Appearance()) -> None:
        capture = None
        try:
            with tempfile.NamedTemporaryFile(prefix="ppe-input-", suffix=suffix, delete=False) as handle:
                self.path = Path(handle.name)
                handle.write(data)
            del data
            capture = cv2.VideoCapture(str(self.path))
            if not capture.isOpened():
                raise ValueError("Cannot open video. Try an H.264 MP4 file.")
            start = perf_counter()
            while not self.stop_event.is_set():
                with self.lock:
                    stale = monotonic() - self.last_poll > 30
                if stale:
                    break
                ok, frame = capture.read()
                if not ok:
                    if self.frames == 0:
                        raise ValueError("The video contains no decodable frames.")
                    break
                with self.lock:
                    appearance = self.appearance
                result = detector.predict(frame, settings, appearance)
                with self.lock:
                    self.result = result
                    self.frames += 1
                    self.fps = self.frames / max(perf_counter() - start, 0.001)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            if capture is not None:
                capture.release()
            try:
                if self.path is not None:
                    self.path.unlink(missing_ok=True)
            except OSError as exc:
                with self.lock:
                    self.error = f"Temporary upload cleanup failed: {exc}"
            self.done.set()


def resize_video_preview(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit a video preview inside the given bounds without stretching or upscaling."""
    if width < 1 or height < 1:
        raise ValueError("Display width and height must be positive.")
    frame_height, frame_width = frame.shape[:2]
    scale = min(width / frame_width, height / frame_height, 1.0)
    if scale == 1.0:
        return frame
    return cv2.resize(frame, (max(1, round(frame_width * scale)),
                              max(1, round(frame_height * scale))), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local webcam/video inference; q or Esc stops playback.")
    parser.add_argument("--source", default="0", help="Camera index or local video path")
    parser.add_argument("--show", action="store_true", help="Open a GUI window (requires non-headless OpenCV)")
    parser.add_argument("--display-width", type=int, default=1280, help="Maximum video preview width (default: 1280)")
    parser.add_argument("--display-height", type=int, default=720, help="Maximum video preview height (default: 720)")
    add_inference_arguments(parser)
    args = parser.parse_args()
    if args.display_width < 1 or args.display_height < 1:
        parser.error("Display width and height must be positive.")
    capture = None
    try:
        settings = Settings(args.conf, args.iou, args.imgsz)
        detector = load_model(args.model)
        source = int(args.source) if args.source.isdecimal() else str(Path(args.source).expanduser().resolve())
        if isinstance(source, str) and not Path(source).is_file():
            raise FileNotFoundError(source)
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError("Cannot open source. Check camera permissions, index, or video codec.")
        frames, start = 0, perf_counter()
        while True:
            ok, frame = capture.read()
            if not ok:
                if isinstance(source, int) or frames == 0:
                    raise RuntimeError("Camera disconnected or source has no decodable frames.")
                break
            result = detector.predict(frame, settings)
            frames += 1
            print(f"\rFrames: {frames} | FPS: {frames / (perf_counter() - start):.1f}", end="", flush=True)
            if args.show:
                preview = result.image_bgr
                if isinstance(source, str):
                    preview = resize_video_preview(preview, args.display_width, args.display_height)
                cv2.imshow("PPE Detection", preview)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")) or cv2.getWindowProperty("PPE Detection", cv2.WND_PROP_VISIBLE) < 1:
                    break
        print()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:
        parser.exit(1, f"\nError: {exc}\nGUI playback requires opencv-python instead of opencv-python-headless.\n")
    finally:
        if capture is not None:
            capture.release()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()

"""Streamlit PPE dashboard. Launch with streamlit run app.py."""
from dataclasses import asdict

import streamlit as st

from predict import DEFAULT_MODEL, Settings, load_model, read_image
from visualization import Appearance


@st.cache_resource
def cached_detector(path: str, modified_ns: int):
    """Cache one synchronized detector per checkpoint revision."""
    return load_model(path)


def main() -> None:
    st.set_page_config(page_title="PPE Detection", page_icon="🦺", layout="wide")
    st.title("🦺 PPE Detection")
    st.caption("Visual safety inspection · YOLO11 · Image detection")
    with st.sidebar:
        st.header("Detection controls")
        settings = Settings(st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05),
                            st.slider("IoU threshold", 0.0, 1.0, 0.45, 0.05),
                            st.select_slider("Image size", options=[320, 480, 640, 960, 1280], value=640))
        st.caption("Higher resolution may improve small-object detection but requires more compute.")
        st.header("Detection Appearance")
        appearance = Appearance(
            st.slider("Bounding Box Thickness", 2, 8, 5),
            st.slider("Label Font Size", 0.5, 2.0, 1.0, 0.1),
            st.toggle("Show Confidence", value=True),
            st.toggle("Show Class Labels", value=True),
        )
        st.caption("Sizes scale with frame width; values refer to 1280px.")
    try:
        if not DEFAULT_MODEL.is_file():
            raise FileNotFoundError(f"Place your trained checkpoint at {DEFAULT_MODEL}. No model is downloaded automatically.")
        with st.spinner("Loading PPE model…"):
            detector = cached_detector(str(DEFAULT_MODEL), DEFAULT_MODEL.stat().st_mtime_ns)
    except Exception as exc:
        st.error(f"Cannot load model: {exc}")
        st.stop()
    st.sidebar.success(f"Device: {detector.device}")
    st.sidebar.caption("Classes: " + ", ".join(detector.names.values()))

    upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if upload is not None and st.button("Detect PPE", type="primary"):
        try:
            with st.spinner("Analyzing image…"):
                result = detector.predict(read_image(upload), settings, appearance)
            st.image(result.image_bgr, channels="BGR", caption="Detection results", use_container_width=True)
            left, right = st.columns(2)
            left.metric("Detections", len(result.detections))
            right.metric("Inference time", f"{result.inference_ms:.1f} ms")
            st.dataframe([{"Class": name, "Count": count} for name, count in result.counts.items()], hide_index=True)
            if result.detections:
                st.dataframe([asdict(d) for d in result.detections], hide_index=True, use_container_width=True)
            else:
                st.info("No detections at the selected threshold.")
        except Exception as exc:
            st.error(f"Image processing failed: {exc}")

    st.divider()
    st.caption("Decision support only. Detections can be missed or incorrect; human safety supervision is required.")


if __name__ == "__main__":
    main()

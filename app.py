"""Streamlit app: receipt (bonnetje) segmentation with YOLO11n-seg."""
import pandas as pd
import streamlit as st

import inference as inf

st.set_page_config(page_title="Receipt Segmentation · YOLO11n-seg",
                   page_icon="🧾", layout="wide")


@st.cache_resource(show_spinner="Loading model (first run downloads ~6 MB)…")
def get_model():
    return inf.load_model()


def run_and_show(model, image_rgb, name, conf, iou, imgsz):
    with st.spinner("Running segmentation…"):
        res = inf.predict(model, image_rgb, conf=conf, iou=iou, imgsz=imgsz)

    c1, c2 = st.columns(2)
    c1.subheader("Original")
    c1.image(image_rgb, width="stretch")
    c2.subheader("Segmentation")
    c2.image(res.annotated_rgb, width="stretch")

    if not res.detections:
        st.warning("No receipt detected. Try lowering the confidence threshold, "
                   "or use a photo where the receipt is clearly visible.")
        return

    st.success(f"Found {len(res.detections)} receipt(s) · {res.inference_ms:.0f} ms")
    st.dataframe(pd.DataFrame([{
        "#": d.index, "class": d.label, "confidence": round(d.confidence, 3),
        "mask area (% of image)": None if d.mask_area_pct is None else round(d.mask_area_pct, 1),
        "box (x1,y1,x2,y2)": tuple(round(v) for v in d.box_xyxy),
    } for d in res.detections]), hide_index=True, width="stretch")

    st.subheader("Extracted receipts")
    deskew = st.checkbox("Straighten (deskew) crops", value=True, key=f"deskew_{name}")
    cols = st.columns(min(len(res.detections), 4))
    crops = []
    for d in res.detections:
        crop = inf.deskew_crop(image_rgb, d) if deskew else inf.crop_box(image_rgb, d)
        crops.append((d, crop))
        with cols[(d.index - 1) % len(cols)]:
            st.image(crop, caption=f"#{d.index} · {d.confidence:.2f}", width="stretch")
            st.download_button(f"Download #{d.index}", inf.to_png_bytes(crop),
                               file_name=f"{name}_receipt_{d.index}.png", mime="image/png",
                               key=f"dl_{name}_{d.index}")
    st.download_button("Download annotated image", inf.to_png_bytes(res.annotated_rgb),
                       file_name=f"{name}_annotated.png", mime="image/png",
                       key=f"dla_{name}")


def main():
    st.title("🧾 Receipt (Bonnetje) Segmentation")
    st.caption("YOLO11n-seg · detects receipts and outlines them with a pixel-level mask.")

    with st.sidebar:
        st.header("Settings")
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        iou = st.slider("IoU (NMS) threshold", 0.10, 0.95, 0.45, 0.05)
        imgsz = st.select_slider("Inference image size", options=[320, 416, 512, 640, 800, 960],
                                 value=640)
        st.markdown("---")
        st.markdown("Model: [`ciCic/yolo11n-seg-bonnetjes`]"
                    "(https://huggingface.co/ciCic/yolo11n-seg-bonnetjes)")

    try:
        model = get_model()
    except Exception as e:  # network / hub / corrupt weights
        st.error("Could not load the model. If this is the first run, check the internet "
                 "connection, or place your weights at `models/best.pt`.")
        st.exception(e)
        st.stop()

    tab_up, tab_cam = st.tabs(["📁 Upload images", "📷 Camera"])

    with tab_up:
        files = st.file_uploader("Upload one or more images",
                                 type=["jpg", "jpeg", "png", "webp", "bmp"],
                                 accept_multiple_files=True)
        for i, f in enumerate(files or []):
            st.markdown(f"### {f.name}")
            try:
                img = inf.load_image(f)
            except Exception:
                st.error(f"Could not read **{f.name}** as an image.")
                continue
            run_and_show(model, img, f"{i}_{f.name.rsplit('.', 1)[0]}", conf, iou, imgsz)
            st.divider()
        if not files:
            st.info("Upload a photo of a receipt to get started.")

    with tab_cam:
        shot = st.camera_input("Take a photo of a receipt")
        if shot is not None:
            run_and_show(model, inf.load_image(shot), "camera", conf, iou, imgsz)


if __name__ == "__main__":
    main()

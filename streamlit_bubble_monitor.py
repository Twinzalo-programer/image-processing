# app.py
# Requires: pip install streamlit opencv-python-headless scikit-image numpy matplotlib
import streamlit as st
import cv2
import numpy as np
from skimage import measure, morphology, segmentation, filters, color
from matplotlib import pyplot as plt

st.title("Froth Image Analyzer — quick heuristic recommendations")

uploaded = st.file_uploader("Upload froth image (jpg/png)", type=['jpg','jpeg','png'])
if uploaded:
    # read image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Input image", use_column_width=True)

    # convert and smooth
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # adaptive threshold (handles uneven illumination)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, 7)

    # morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # label connected components
    labels = measure.label(clean, connectivity=2)
    props = measure.regionprops(labels)
    areas = [p.area for p in props if p.area>20]  # filter tiny noise
    total_area = np.sum(areas)
    image_area = img.shape[0]*img.shape[1]
    coverage_pct = 100.0 * total_area / image_area if image_area>0 else 0.0
    mean_area = np.mean(areas) if len(areas)>0 else 0
    n_bubbles = len(areas)

    # simple watershed to cleanly separate touching bubbles (optional)
    # (skipped for brevity; can be added later)

    st.write(f"Detected bubbles: **{n_bubbles}**")
    st.write(f"Surface coverage (approx): **{coverage_pct:.1f}%**")
    st.write(f"Mean bubble area (pixels): **{mean_area:.1f}**")

    # heuristic recommendations (example rules)
    recs = []
    if coverage_pct < 8:
        recs.append("Low froth coverage → Increase aeration or reduce wash water/air disturbance.")
    elif coverage_pct > 45:
        recs.append("High froth coverage → Possibly too much frother or too much air; check concentrate entrainment.")
    else:
        recs.append("Coverage in typical range — monitor trend, not just single image.")

    if mean_area > 2000 and n_bubbles < 30:
        recs.append("Few large bubbles → possible collector overdosage or poor frother balance; consider reducing collector or adjusting frother.")
    elif mean_area < 200 and n_bubbles>200:
        recs.append("Many small bubbles → might indicate excessive frother or high shear; consider reducing frother or aeration rate.")

    # present recommendations
    st.header("Automated recommendations (heuristic)")
    for r in recs:
        st.info(r)

    # show segmentation
    fig, ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].imshow(img_rgb); ax[0].set_title("Original"); ax[0].axis('off')
    ax[1].imshow(clean, cmap='gray'); ax[1].set_title("Segmentation"); ax[1].axis('off')
    st.pyplot(fig)


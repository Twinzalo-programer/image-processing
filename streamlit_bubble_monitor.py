
# streamlit_bubble_monitor.py
import streamlit as st
import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Bubble Size Analyzer", layout="centered")
st.title("🫧 Bubble Size Analyzer - Photo Upload")

st.markdown("""
Upload a froth flotation image (JPG or PNG). The app will detect bubbles, calculate their diameters,
and display the results including a histogram and downloadable CSV.
""")

# Calibration input
microns_per_pixel = st.number_input("Microns per Pixel (µm/pixel)", min_value=1.0, max_value=1000.0, value=50.0)

# File upload
uploaded_file = st.file_uploader("Upload Froth Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=50, param2=30, minRadius=10, maxRadius=50
    )

    bubble_data = []
    output = image.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            diameter_px = 2 * r
            diameter_um = diameter_px * microns_per_pixel
            bubble_data.append(diameter_um)
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.rectangle(output, (x-2, y-2), (x+2, y+2), (0, 128, 255), -1)

    st.subheader("📷 Detected Bubbles")
    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), channels="RGB")

    if bubble_data:
        st.subheader("📊 Bubble Diameter Histogram")
        fig, ax = plt.subplots()
        ax.hist(bubble_data, bins=10, edgecolor='black', color='skyblue')
        ax.set_xlabel("Diameter (µm)")
        ax.set_ylabel("Count")
        ax.set_title("Bubble Diameter Distribution")
        st.pyplot(fig)

        st.write(f"**Average Diameter:** {np.mean(bubble_data):.2f} µm")
        st.write(f"**Bubble Count:** {len(bubble_data)}")

        df = pd.DataFrame({"Bubble Diameter (µm)": bubble_data})
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "bubble_diameters.csv", "text/csv")
    else:
        st.warning("No bubbles detected. Try another image or adjust resolution.")

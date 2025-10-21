import streamlit as st
import cv2
import numpy as np
import json
import os
import base64
from io import BytesIO
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ------------------------- Config / Colors --------------------------
API_COLOR_REFERENCES = {
    "ph": {"7.4": (222,222,222), "7.8": (237,222,192), "8.0": (229,203,186),
           "8.2": (192,141,140), "8.4": (127,74,127), "8.8": (126,101,144)},
    "ammonia": {"0 ppm": (212,212,212), "0.25 ppm": (241,237,188),
                "0.5 ppm": (223,228,171), "1.0 ppm": (185,214,139),
                "2.0 ppm": (85,166,80), "4.0 ppm": (86,173,122),
                "8.0 ppm": (147,174,158)},
    "nitrite": {"0 ppm": (227,227,227), "0.25 ppm": (219,228,233),
                "0.5 ppm": (192,180,197), "1.0 ppm": (161,130,165),
                "2.0 ppm": (130,63,117), "5.0 ppm": (140,92,131)},
    "nitrate": {"0 ppm": (225,226,226), "5 ppm": (239,230,179),
                "10 ppm": (228,189,166), "20 ppm": (234,163,129),
                "40 ppm": (204,67,65), "80 ppm": (203,97,98),
                "160 ppm": (190,143,145)}
}
ORDERED_TESTS = ["ph", "ammonia", "nitrite", "nitrate"]

# ------------------------- Helpers ---------------------------------
def simple_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    imgf = img_bgr.astype(np.float32)
    avg = np.mean(imgf, axis=(0, 1))
    gray = float(np.mean(avg))
    imgf *= gray / (avg + 1e-6)
    return np.clip(imgf, 0, 255).astype(np.uint8)

def rgb_to_lab(rgb):
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0]

def color_distance(c1, c2):
    return np.linalg.norm(np.array(c1, float) - np.array(c2, float))

def match_api_color(avg_rgb, ref_dict):
    lab = rgb_to_lab(avg_rgb)
    return min(ref_dict.items(), key=lambda kv: color_distance(lab, rgb_to_lab(kv[1])))[0]

def median_color_rgb(bgr_roi: np.ndarray):
    sel = bgr_roi.reshape(-1, 3)
    r, g, b = np.median(sel[:, 2]), np.median(sel[:, 1]), np.median(sel[:, 0])
    return int(r), int(g), int(b)

def make_canvas_bg_pil(src_pil: Image.Image, max_w=900, max_h=700) -> tuple[Image.Image, int, int]:
    """
    Resize the PIL image to fit within (max_w, max_h) while preserving aspect ratio,
    then return (resized_pil, canvas_w, canvas_h). We pass this *pre-sized PIL* directly
    to st_canvas(background_image=...), which avoids the library's internal resize path.
    """
    ow, oh = src_pil.width, src_pil.height
    scale = min(max_w / ow, max_h / oh, 1.0)  # don't upscale
    cw, ch = int(ow * scale), int(oh * scale)
    # Use LANCZOS for quality; size must match exactly what we pass to st_canvas height/width
    bg = src_pil.resize((cw, ch), Image.LANCZOS)
    return bg, cw, ch

# ------------------------- App -------------------------------------
st.set_page_config(page_title="API Test Analyzer", layout="centered")
st.title("🧪 Aquarium Water Test Analyzer")

st.markdown(
    """
Upload your aquarium test kit image and **draw rectangles** over the four regions:
**pH**, **Ammonia**, **Nitrite**, and **Nitrate**.  
Works on desktop 💻 and mobile 📱.
"""
)

uploaded = st.file_uploader("📸 Upload test image",
                            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])

if uploaded:
    # Load original as RGB PIL
    orig_pil = Image.open(uploaded).convert("RGB")

    # White balance for *analysis only* (we don't need to show this on the canvas)
    img_bgr = cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2BGR)
    img_bgr = simple_white_balance(img_bgr)

    st.subheader("Step 1 – Draw 4 regions (pH, Ammonia, Nitrite, Nitrate)")
    st.markdown("👉 Use the **rectangle tool** to draw each region in order (left-to-right or top-to-bottom).")

    # Build a background image *already sized* to the canvas dimensions.
    bg_pil, canvas_w, canvas_h = make_canvas_bg_pil(orig_pil, max_w=700, max_h=600)
    bg_array = np.array(bg_pil)
    
    # CSS to hide white space
    st.markdown(f"""
        <style>
        .stApp {{
            max-width: 1200px;
        }}
        div[data-testid="stVerticalBlock"] > div:has(iframe) {{
            width: {bg_array.shape[1]}px !important;
            max-width: {bg_array.shape[1]}px !important;
        }}
        </style>
    """, unsafe_allow_html=True)

    # Draw the canvas with numpy array background - use exact image dimensions
    canvas_result = st_canvas(
        fill_color="rgba(255,0,0,0.15)",
        stroke_color="#FF0000",
        stroke_width=3,
        background_image=bg_array,
        update_streamlit=True,
        height=bg_array.shape[0],  # Use exact height from the array
        width=bg_array.shape[1],   # Use exact width from the array
        drawing_mode="rect",
        key="canvas",
    )

    # --- Analyze on click ---
    if st.button("🔍 Analyze Selected Regions"):
        if canvas_result.json_data is None or len(canvas_result.json_data.get("objects", [])) < 4:
            st.error("Please draw **exactly 4 rectangles** (pH, Ammonia, Nitrite, Nitrate).")
        else:
            # Map canvas coordinates (on resized image) back to original / to the white-balanced BGR for analysis
            scale_x = img_bgr.shape[1] / bg_array.shape[1]
            scale_y = img_bgr.shape[0] / bg_array.shape[0]

            rects = canvas_result.json_data["objects"][:4]
            results = {}
            annotated = img_bgr.copy()

            for i, test in enumerate(ORDERED_TESTS):
                left = int(rects[i]["left"] * scale_x)
                top = int(rects[i]["top"] * scale_y)
                width = int(rects[i]["width"] * scale_x)
                height = int(rects[i]["height"] * scale_y)

                # Clamp to image bounds
                left = max(0, min(left, annotated.shape[1]-1))
                top = max(0, min(top, annotated.shape[0]-1))
                width = max(1, min(width, annotated.shape[1]-left))
                height = max(1, min(height, annotated.shape[0]-top))

                roi = annotated[top:top+height, left:left+width]
                avg_rgb = median_color_rgb(roi)  # returns (R,G,B)
                color_name = match_api_color(avg_rgb, API_COLOR_REFERENCES[test])

                results[test] = {"color_name": color_name, "rgb": avg_rgb}

                # Annotate (draw rectangles/labels) on BGR image
                cv2.rectangle(annotated, (left, top), (left+width, top+height), (255, 255, 255), 2)
                label = f"{test.upper()} = {color_name.replace('ppm','').strip()}"
                cv2.putText(annotated, label, (left, max(40, top-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

            st.subheader("Step 2 – Results")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected Regions", width=None)
            st.json(results)

            # Download JSON
            result_data = {"file": uploaded.name}
            for t in ORDERED_TESTS:
                if t in results:
                    result_data[t] = results[t]["color_name"]
                    result_data[f"{t}_RGB"] = results[t]["rgb"]
            json_str = json.dumps(result_data, indent=2)
            st.download_button(
                "📥 Download JSON Results",
                json_str,
                file_name=f"{os.path.splitext(uploaded.name)[0]}_results.json",
                mime="application/json"
            )
else:
    st.info("Upload an image above to start.")

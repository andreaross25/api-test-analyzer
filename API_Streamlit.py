import streamlit as st
import cv2
import numpy as np
import json
import os
import base64
from io import BytesIO
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ---------- 🧩 Monkey-patch for image_to_url ----------
import streamlit.elements.image as st_image

def safe_image_to_url(image, width=None, clamp=False,
                      channels="RGB", output_format="auto", *args, **kwargs):
    """Compatibility patch for streamlit-drawable-canvas"""
    if isinstance(image, Image.Image):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        return "data:image/png;base64," + base64.b64encode(data).decode()
    elif isinstance(image, np.ndarray):
        _, buffer = cv2.imencode(".png", image[:, :, ::-1])
        return "data:image/png;base64," + base64.b64encode(buffer).decode()
    elif isinstance(image, str) and image.startswith("data:image"):
        return image
    return None

st_image.image_to_url = safe_image_to_url
# ------------------------------------------------------

# ---------- API reference colors ----------
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
ORDERED_TESTS = ["ph","ammonia","nitrite","nitrate"]

# ---------- Helper functions ----------
def simple_white_balance(img):
    imgf = img.astype(np.float32)
    avg = np.mean(imgf, axis=(0,1))
    gray = np.mean(avg)
    imgf *= gray / (avg + 1e-6)
    return np.clip(imgf,0,255).astype(np.uint8)

def rgb_to_lab(rgb):
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0,0]

def color_distance(c1,c2):
    return np.linalg.norm(np.array(c1,float)-np.array(c2,float))

def match_api_color(avg_rgb, ref_dict):
    lab = rgb_to_lab(avg_rgb)
    return min(ref_dict.items(),
               key=lambda kv: color_distance(lab, rgb_to_lab(kv[1])))[0]

def median_color_rgb(bgr_roi):
    sel = bgr_roi.reshape(-1,3)
    r,g,b = np.median(sel[:,2]), np.median(sel[:,1]), np.median(sel[:,0])
    return int(r),int(g),int(b)

# ---------- Streamlit App ----------
st.set_page_config(page_title="API Test Analyzer", layout="wide")
st.title("🧪 Aquarium Water Test Analyzer")

st.markdown("""
Upload your aquarium test kit image and **draw rectangles** over the four regions:
**pH**, **Ammonia**, **Nitrite**, and **Nitrate**.  
Works seamlessly on both desktop 💻 and mobile 📱.
""")

uploaded = st.file_uploader("📸 Upload test image",
                            type=["jpg","jpeg","png","bmp","tif","tiff","webp"])

if uploaded:
    # --- Load & preprocess ---
    img_pil = Image.open(uploaded).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_bgr = simple_white_balance(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil_fixed = Image.fromarray(img_rgb)   # ✅ PIL image after white balance

    st.image(img_rgb, caption="Uploaded Image", width=None)

    st.subheader("Step 1 – Draw 4 regions (pH, Ammonia, Nitrite, Nitrate)")
    st.markdown("👉 Use the **rectangle tool** to draw each region in order.")

    # ---------- Drawable Canvas ----------
    canvas_result = st_canvas(
        fill_color="rgba(0,255,0,0.1)",
        stroke_color="red",
        stroke_width=2,
        background_image=img_pil_fixed,   # ✅ use PIL Image to avoid NumPy truth value issue
        update_streamlit=True,
        height=min(700, img_pil_fixed.height),
        width=min(900, img_pil_fixed.width),
        drawing_mode="rect",
        key="canvas",
    )

    # ---------- Analyze ----------
    if st.button("🔍 Analyze Selected Regions"):
        if canvas_result.json_data is None or len(canvas_result.json_data["objects"]) < 4:
            st.error("Please draw **exactly 4 rectangles** (pH, Ammonia, Nitrite, Nitrate).")
        else:
            rects = canvas_result.json_data["objects"][:4]
            results = {}
            annotated = img_bgr.copy()

            for i, test in enumerate(ORDERED_TESTS):
                left  = int(rects[i]["left"])
                top   = int(rects[i]["top"])
                width = int(rects[i]["width"])
                height= int(rects[i]["height"])

                roi = annotated[top:top+height, left:left+width]
                avg_rgb = median_color_rgb(roi)
                color_name = match_api_color(avg_rgb, API_COLOR_REFERENCES[test])
                results[test] = {"color_name": color_name, "rgb": avg_rgb}

                cv2.rectangle(annotated,(left,top),(left+width,top+height),(255,255,255),2)
                label = f"{test.upper()} = {color_name.replace('ppm','').strip()}"
                cv2.putText(annotated,label,(left,max(40,top-10)),
                            cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2,cv2.LINE_AA)

            st.subheader("Step 2 – Results")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Detected Regions", width=None)
            st.json(results)

            # Save JSON
            result_data = {"file": uploaded.name}
            for t in ORDERED_TESTS:
                if t in results:
                    result_data[t] = results[t]["color_name"]
                    result_data[f"{t}_RGB"] = results[t]["rgb"]
            json_str = json.dumps(result_data, indent=2)
            st.download_button("📥 Download JSON Results",
                               json_str,
                               file_name=f"{os.path.splitext(uploaded.name)[0]}_results.json",
                               mime="application/json")
else:
    st.info("Upload an image above to start.")
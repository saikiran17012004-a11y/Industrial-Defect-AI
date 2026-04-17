import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# 1. Page Configuration
st.set_page_config(page_title="AI Defect Inspector", layout="wide")
st.title("🏭 Project 1: Part Defect Detection & Segregation")
st.markdown("---")

# 2. Load the Model Safely
# Update this path if your best.pt is in a different folder
MODEL_PATH = "best.pt" 

@st.cache_resource
def load_trained_model(path):
    if os.path.exists(path):
        return YOLO(path)
    else:
        # Fallback to base model if best.pt is missing
        return YOLO("yolov8n.pt")

model = load_trained_model(MODEL_PATH)

# 3. Sidebar Configuration
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# 4. Image Upload Section
uploaded_file = st.file_uploader("Upload an industrial part image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image using PIL
    image = Image.open(uploaded_file)
    
    # Create two columns for UI
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("AI Inspection Result")
        
        with st.spinner("Analyzing surface for defects..."):
            # Convert PIL image to NumPy array for YOLO consistency
            img_array = np.array(image)
            
            # RUN PREDICTION
            results = model.predict(img_array, conf=conf_threshold)
            
            # Show processed image with boxes
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Detection Visualization", use_column_width=True)
            
            # 5. Segregation Logic
            num_defects = len(results[0].boxes)
            
            if num_defects > 0:
                st.error(f"❌ STATUS: DEFECTIVE ({num_defects} issues found)")
                st.warning("ACTION: Segregate to REJECTION BIN B")
                st.button("Confirm Rejection")
            else:
                st.success("✅ STATUS: OK")
                st.info("ACTION: Proceed to PACKAGING LINE")
                st.button("Confirm Pass")

else:
    st.info("Please upload an image to start the automated inspection.")
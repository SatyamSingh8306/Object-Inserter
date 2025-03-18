import os
import streamlit as st
import numpy as np
import torch
from PIL import Image
import tempfile
from ai import ProductImageIntegrator

# Set page config
st.set_page_config(
    page_title="Product Image Integrator",
    page_icon="🖼️",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Function to save uploaded files to temp directory
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        return temp_file.name

# Main app header
st.title("Product Image Integration Tool")
st.markdown("Upload a product image and a background scene to seamlessly integrate them together.")

# Device selection
device_options = ["cpu"]
if torch.cuda.is_available():
    device_options.append("cuda")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device_options.append("mps")

selected_device = st.sidebar.selectbox("Processing Device", device_options)

# Initialize the processor
@st.cache_resource
def get_processor(device):
    with st.spinner("Loading AI models (this may take a while)..."):
        return ProductImageIntegrator(device=device)

# Load processor based on device
try:
    processor = get_processor(selected_device)
    st.sidebar.success(f"Models loaded successfully on {selected_device}")
except Exception as e:
    st.sidebar.error(f"Error loading models: {str(e)}")
    st.stop()

# Create two columns for the UI layout
col1, col2 = st.columns(2)

# First column for inputs
with col1:
    st.header("Input Images")
    
    # Product image upload
    product_file = st.file_uploader("Upload Product Image", type=["png", "jpg", "jpeg"])
    if product_file is not None:
        product_image = Image.open(product_file)
        st.image(product_image, caption="Product Image", use_column_width=True)
        product_path = save_uploaded_file(product_file)
    
    # Background image upload
    background_file = st.file_uploader("Upload Background Scene", type=["png", "jpg", "jpeg"])
    if background_file is not None:
        background_image = Image.open(background_file)
        st.image(background_image, caption="Background Scene", use_column_width=True)
        background_path = save_uploaded_file(background_file)
    
    # Process button
    if product_file is not None and background_file is not None:
        if st.button("Integrate Product into Scene"):
            with st.spinner("Processing images... This may take a few minutes."):
                try:
                    result = processor.process_image_pair(product_path, background_path)
                    st.session_state.processed_image = result
                    st.session_state.processing_complete = True
                    # Clean up temp files
                    os.unlink(product_path)
                    os.unlink(background_path)
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")

# Second column for output
with col2:
    st.header("Processed Result")
    if st.session_state.processing_complete and st.session_state.processed_image is not None:
        st.image(st.session_state.processed_image, caption="Integrated Result", use_column_width=True)
        
        # Add download button for the processed image
        result_img = st.session_state.processed_image
        if result_img:
            # Convert PIL image to bytes
            buf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            result_img.save(buf.name, format="PNG")
            with open(buf.name, 'rb') as f:
                img_bytes = f.read()
            os.unlink(buf.name)
            
            st.download_button(
                label="Download Result",
                data=img_bytes,
                file_name="integrated_product.png",
                mime="image/png"
            )
    else:
        st.info("Your processed image will appear here.")

# Advanced settings in sidebar
st.sidebar.header("Advanced Settings")

with st.sidebar.expander("Placement Settings"):
    placement_method = st.selectbox(
        "Placement Method",
        ["Automatic", "Manual"],
        help="Choose how to position the product in the scene"
    )
    
    if placement_method == "Manual":
        st.text("Manual placement options:")
        placement_x = st.slider("X Position (%)", 0, 100, 50)
        placement_y = st.slider("Y Position (%)", 0, 100, 50)
        scale = st.slider("Scale", 0.1, 2.0, 1.0)

with st.sidebar.expander("Lighting Settings"):
    lighting_match = st.slider(
        "Lighting Match Strength", 
        0.0, 1.0, 0.7,
        help="How strongly to match the background lighting"
    )

with st.sidebar.expander("Batch Processing"):
    st.text("Upload multiple files for batch processing")
    batch_products = st.file_uploader("Upload Multiple Products", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    batch_backgrounds = st.file_uploader("Upload Multiple Backgrounds", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if batch_products and batch_backgrounds:
        if st.button("Process Batch"):
            st.text(f"Will process {len(batch_products)} products with {len(batch_backgrounds)} backgrounds")
            # Implementation for batch processing would go here

# About section
st.sidebar.header("About")
st.sidebar.info(
    """
    This tool uses AI to seamlessly integrate product images into background scenes.
    
    Features:
    - Automatic background removal
    - Scene analysis for optimal placement
    - Lighting adjustment
    - Realistic blending using generative AI
    """
)
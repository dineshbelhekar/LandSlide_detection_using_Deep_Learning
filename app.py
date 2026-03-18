import streamlit as st
import numpy as np
import h5py
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import io
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Landslide Detection System",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🌍 Landslide Detection System</h1><p>Using U-Net for Semantic Segmentation of Satellite Imagery</p></div>', 
            unsafe_allow_html=True)

# Load model (cached for faster UI)
@st.cache_resource
def load_landslide_model():
    """Load the trained U-Net model"""
    try:
        # Load with custom objects if you used custom loss/metrics
        model = load_model("best_model.h5", compile=False)
        # Recompile if needed
        model.compile(optimizer='adam', loss='binary_crossentropy', 
                     metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def normalize_for_display(image_data):
    """Normalize image data to [0, 1] range for display"""
    # Handle NaN values
    image_data = np.nan_to_num(image_data, nan=0.0)
    
    # Get min and max
    min_val = image_data.min()
    max_val = image_data.max()
    
    # Avoid division by zero
    if max_val - min_val == 0:
        return np.zeros_like(image_data)
    
    # Normalize to [0, 1]
    normalized = (image_data - min_val) / (max_val - min_val)
    return normalized

def create_rgb_image(data):
    """Create RGB image from bands 3,2,1 (R,G,B) and normalize for display"""
    # Extract RGB bands
    rgb = np.stack([
        data[:, :, 3],  # Red
        data[:, :, 2],  # Green
        data[:, :, 1]   # Blue
    ], axis=-1)
    
    # Normalize for display
    rgb_normalized = normalize_for_display(rgb)
    return rgb_normalized

def preprocess_for_model(data):
    """
    Preprocess the H5 image data according to the training pipeline
    Args:
        data: numpy array from H5 file with shape (128, 128, 14)
    Returns:
        preprocessed array of shape (1, 128, 128, 6)
    """
    data = data.copy()
    
    # Replace NaN values
    data[np.isnan(data)] = 0.000001
    
    # Calculate normalization factors
    mid_rgb = data[:, :, 1:4].max() / 2.0
    mid_slope = data[:, :, 12].max() / 2.0
    mid_elevation = data[:, :, 13].max() / 2.0
    
    # NDVI calculation (Normalized Difference Vegetation Index)
    data_red = data[:, :, 3]  # Band 3 is Red
    data_nir = data[:, :, 7]  # Band 7 is NIR
    # Avoid division by zero
    denominator = data_nir + data_red
    denominator[denominator == 0] = 0.000001
    data_ndvi = (data_nir - data_red) / denominator
    
    # Create feature array (6 channels)
    features = np.zeros((128, 128, 6))
    features[:, :, 0] = 1 - data[:, :, 3] / mid_rgb  # RED (normalized)
    features[:, :, 1] = 1 - data[:, :, 2] / mid_rgb  # GREEN (normalized)
    features[:, :, 2] = 1 - data[:, :, 1] / mid_rgb  # BLUE (normalized)
    features[:, :, 3] = data_ndvi  # NDVI
    features[:, :, 4] = 1 - data[:, :, 12] / mid_slope  # SLOPE (band 12)
    features[:, :, 5] = 1 - data[:, :, 13] / mid_elevation  # ELEVATION (band 13)
    
    # Clip values to reasonable range
    features = np.clip(features, -1, 1)
    
    # Add batch dimension
    features = np.expand_dims(features, axis=0)
    
    return features

def calculate_landslide_metrics(pred_mask):
    """Calculate landslide metrics from prediction mask"""
    total_pixels = pred_mask.size
    landslide_pixels = np.sum(pred_mask > 0.5)
    percentage = (landslide_pixels / total_pixels) * 100
    
    # Determine severity
    if percentage > 30:
        severity = "🔴 HIGH"
        severity_color = "red"
    elif percentage > 10:
        severity = "🟡 MEDIUM"
        severity_color = "orange"
    elif percentage > 0:
        severity = "🟢 LOW"
        severity_color = "green"
    else:
        severity = "⚪ NONE"
        severity_color = "gray"
    
    return {
        'percentage': percentage,
        'severity': severity,
        'severity_color': severity_color,
        'has_landslide': landslide_pixels > 0
    }

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/landslide.png", width=100)
    st.header("About")
    st.info("""
    This system uses a U-Net deep learning model trained on the Landslide4Sense dataset 
    to detect landslides in multi-spectral satellite imagery.
    
    **Features used:**
    - RGB bands (1-3)
    - NDVI (Vegetation index)
    - Slope
    - Elevation
    """)
    
    st.header("Instructions")
    st.markdown("""
    1. Upload an H5 file containing satellite imagery
    2. Wait for processing
    3. View the landslide prediction mask
    4. Check severity assessment
    """)
    
    # Model info
    model = load_landslide_model()
    if model is not None:
        st.success("✅ Model loaded successfully")
    else:
        st.error("❌ Model not loaded")

# Main content
if model is None:
    st.error("Failed to load model. Please check if 'best_model.h5' exists in the current directory.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "📤 Upload H5 Image File", 
    type=["h5"],
    help="Upload a .h5 file containing satellite imagery (14 bands, 128x128 pixels)"
)

if uploaded_file is not None:
    try:
        # Read H5 file
        with h5py.File(uploaded_file, 'r') as hdf:
            # Check available keys
            keys = list(hdf.keys())
            st.write(f"📋 Available datasets: {keys}")
            
            # Get the image data (assuming 'img' key exists)
            if 'img' in keys:
                img_data = np.array(hdf.get('img'))
            else:
                # Try to find any dataset that looks like image data
                for key in keys:
                    data = np.array(hdf.get(key))
                    if len(data.shape) == 3 and data.shape[2] in [14, 3, 4]:
                        img_data = data
                        st.info(f"Using dataset: {key}")
                        break
                else:
                    st.error("No suitable image dataset found in H5 file")
                    st.stop()
        
        # Display file info
        st.write(f"📊 Image shape: {img_data.shape}")
        
        # Create two columns for display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original RGB Image")
            # Create RGB visualization with proper normalization
            if img_data.shape[2] >= 4:  # Has at least RGB bands
                rgb_img = create_rgb_image(img_data)
                st.image(rgb_img, caption="RGB Composite (normalized)", use_column_width=True)
                
                # Show image statistics
                with st.expander("📊 Image Statistics"):
                    st.write(f"Min value: {img_data.min():.4f}")
                    st.write(f"Max value: {img_data.max():.4f}")
                    st.write(f"Mean value: {img_data.mean():.4f}")
                    st.write(f"Data type: {img_data.dtype}")
            else:
                st.warning("Cannot create RGB image - insufficient bands")
                # Try to display first 3 bands with normalization
                first_bands = img_data[:, :, :3]
                normalized_bands = normalize_for_display(first_bands)
                st.image(normalized_bands, caption="First 3 bands (normalized)", use_column_width=True)
        
        # Preprocess and predict
        with st.spinner("🔄 Processing image and generating prediction..."):
            # Preprocess for model
            features = preprocess_for_model(img_data)
            
            # Predict
            prediction = model.predict(features, verbose=0)
            pred_mask = prediction[0, :, :, 0]  # Remove batch and channel dimensions
        
        with col2:
            st.subheader("🗺️ Predicted Landslide Mask")
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(pred_mask, cmap='RdYlGn_r', vmin=0, vmax=1)
            ax.axis("off")
            plt.colorbar(im, ax=ax, label='Landslide Probability')
            st.pyplot(fig)
            plt.close()
        
        # Calculate and display metrics
        metrics = calculate_landslide_metrics(pred_mask)
        
        # Metrics display
        st.subheader("📊 Analysis Results")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric(
                label="Landslide Area",
                value=f"{metrics['percentage']:.2f}%",
                delta=None
            )
        
        with col4:
            st.markdown(
                f"""
                <div style="padding: 1rem; border-radius: 5px; background-color: {metrics['severity_color']}20; 
                border-left: 4px solid {metrics['severity_color']};">
                    <h4 style="margin: 0; color: {metrics['severity_color']};">Severity Level</h4>
                    <h2 style="margin: 0; color: {metrics['severity_color']};">{metrics['severity']}</h2>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col5:
            st.metric(
                label="Detection Status",
                value="⚠️ Landslide Detected" if metrics['has_landslide'] else "✅ No Landslide",
                delta=None
            )
        
        # Overlay visualization
        st.subheader("🔄 Overlay Visualization")
        fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original RGB (normalized)
        ax1.imshow(rgb_img if img_data.shape[2] >= 4 else normalized_bands)
        ax1.set_title("Original Image")
        ax1.axis("off")
        
        # Prediction mask
        ax2.imshow(pred_mask, cmap='RdYlGn_r', vmin=0, vmax=1)
        ax2.set_title("Landslide Probability")
        ax2.axis("off")
        
        # Overlay
        ax3.imshow(rgb_img if img_data.shape[2] >= 4 else normalized_bands)
        # Create red overlay for high probability areas
        overlay = np.zeros((*pred_mask.shape, 4))
        overlay[pred_mask > 0.5] = [1, 0, 0, 0.5]  # Red with transparency
        ax3.imshow(overlay)
        ax3.set_title("Overlay (Red = Landslide)")
        ax3.axis("off")
        
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        
        # Download results option
        st.subheader("📥 Download Results")
        
        # Save prediction as image
        buf = io.BytesIO()
        plt.imsave(buf, pred_mask, cmap='gray', format='png')
        buf.seek(0)
        
        col6, col7 = st.columns(2)
        with col6:
            st.download_button(
                label="Download Prediction Mask (PNG)",
                data=buf,
                file_name="landslide_prediction.png",
                mime="image/png"
            )
        
        with col7:
            # Create a text report
            import datetime
            report = f"""LANDSLIDE DETECTION REPORT
========================
File: {uploaded_file.name}
Image Shape: {img_data.shape}
Landslide Area: {metrics['percentage']:.2f}%
Severity: {metrics['severity']}
Detection: {'Yes' if metrics['has_landslide'] else 'No'}

Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            st.download_button(
                label="Download Report (TXT)",
                data=report,
                file_name="landslide_report.txt",
                mime="text/plain"
            )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

else:
    # Show example when no file is uploaded
    st.info("👆 Please upload an H5 file to begin analysis")
    
    # Sample visualization of how it works
    st.subheader("How it works")
    col8, col9, col10 = st.columns(3)
    
    with col8:
        st.markdown("**1. Input Processing**")
        st.caption("Multi-spectral satellite image (14 bands)")
    
    with col9:
        st.markdown("**2. Feature Extraction**")
        st.caption("RGB, NDVI, Slope, Elevation")
    
    with col10:
        st.markdown("**3. U-Net Prediction**")
        st.caption("Binary segmentation mask")


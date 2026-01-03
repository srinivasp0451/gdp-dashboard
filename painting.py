import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Learn Portrait Drawing",
    page_icon="✏️",
    layout="wide"
)

# Title and description
st.title("✏️ Learn Realistic Portrait Drawing")
st.markdown("Upload a portrait photo and get guided outlines to practice realistic pencil drawing!")

# Sidebar for controls
st.sidebar.header("Drawing Guide Options")

guide_type = st.sidebar.selectbox(
    "Select Guide Type",
    ["Edge Detection (Contours)", "Simplified Sketch", "Grid Overlay", "Value Map (Shading Guide)", "Combined Guide"]
)

# Guide intensity controls
if guide_type in ["Edge Detection (Contours)", "Simplified Sketch"]:
    edge_threshold1 = st.sidebar.slider("Detail Level (Lower = More Detail)", 30, 150, 50)
    edge_threshold2 = st.sidebar.slider("Edge Strength", 100, 300, 150)

if guide_type == "Grid Overlay":
    grid_size = st.sidebar.slider("Grid Cell Size", 20, 100, 40)

if guide_type == "Value Map (Shading Guide)":
    blur_amount = st.sidebar.slider("Smoothness", 1, 15, 5, step=2)

# Pencil type information
st.sidebar.markdown("---")
st.sidebar.header("📝 Pencil Guide")
st.sidebar.markdown("""
**Light Sketching:**
- 2H, H pencils for initial outlines

**Medium Tones:**
- HB, B, 2B for general shading

**Dark Tones:**
- 4B, 6B for deep shadows

**Darkest Areas:**
- 8B for the darkest parts
""")

# File uploader
uploaded_file = st.file_uploader("Upload a portrait image", type=["jpg", "jpeg", "png", "bmp", "webp"])

def create_edge_detection(img, threshold1, threshold2):
    """Create edge detection guide"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1, threshold2)
    # Invert so edges are black on white
    edges_inverted = cv2.bitwise_not(edges)
    return edges_inverted

def create_simplified_sketch(img, threshold1, threshold2):
    """Create a simplified pencil sketch effect"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert the image
    inverted = cv2.bitwise_not(gray)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    
    # Invert the blurred image
    inverted_blur = cv2.bitwise_not(blurred)
    
    # Create pencil sketch
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    
    # Add edge detection for stronger outlines
    edges = cv2.Canny(gray, threshold1, threshold2)
    
    # Combine sketch with edges
    sketch_with_edges = cv2.bitwise_and(sketch, cv2.bitwise_not(edges))
    
    return sketch_with_edges

def create_grid_overlay(img, grid_size):
    """Create grid overlay for proportions"""
    output = img.copy()
    h, w = output.shape[:2]
    
    # Draw vertical lines
    for x in range(0, w, grid_size):
        cv2.line(output, (x, 0), (x, h), (100, 100, 100), 1)
    
    # Draw horizontal lines
    for y in range(0, h, grid_size):
        cv2.line(output, (0, y), (w, y), (100, 100, 100), 1)
    
    return output

def create_value_map(img, blur_amount):
    """Create shading guide showing light and dark areas"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply blur to simplify values
    blurred = cv2.GaussianBlur(gray, (blur_amount, blur_amount), 0)
    
    # Create posterized version with fewer tones
    posterized = np.floor(blurred / 51) * 51  # Creates 5 tone levels
    posterized = posterized.astype(np.uint8)
    
    return posterized

def create_combined_guide(img, threshold1, threshold2, grid_size):
    """Create a combined guide with edges and grid"""
    # Start with edge detection
    edges = create_edge_detection(img, threshold1, threshold2)
    
    # Convert to BGR for grid overlay
    output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Add grid
    h, w = output.shape[:2]
    for x in range(0, w, grid_size):
        cv2.line(output, (x, 0), (x, h), (200, 100, 100), 1)
    for y in range(0, h, grid_size):
        cv2.line(output, (0, y), (w, y), (200, 100, 100), 1)
    
    return output

def resize_image(img, max_size=800):
    """Resize image to fit display while maintaining aspect ratio"""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

# Main processing
if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Resize for display
    img_resized = resize_image(img)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col2:
        st.subheader(f"Drawing Guide: {guide_type}")
        
        # Generate the selected guide
        if guide_type == "Edge Detection (Contours)":
            result = create_edge_detection(img_resized, edge_threshold1, edge_threshold2)
            st.image(result, use_container_width=True, channels="GRAY")
            st.info("📝 Trace these contour lines with a 2H or H pencil for your initial sketch.")
            
        elif guide_type == "Simplified Sketch":
            result = create_simplified_sketch(img_resized, edge_threshold1, edge_threshold2)
            st.image(result, use_container_width=True, channels="GRAY")
            st.info("📝 Follow this sketch guide. Lighter areas = lighter pencil pressure, darker areas = more pressure.")
            
        elif guide_type == "Grid Overlay":
            result = create_grid_overlay(img_resized, grid_size)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.info("📝 Draw the same grid on your paper. Use it to accurately place features and maintain proportions.")
            
        elif guide_type == "Value Map (Shading Guide)":
            result = create_value_map(img_resized, blur_amount)
            st.image(result, use_container_width=True, channels="GRAY")
            st.info("📝 This shows simplified shading zones. Darkest areas = 6B/8B, Medium = 2B/4B, Light = H/HB")
            
        elif guide_type == "Combined Guide":
            result = create_combined_guide(img_resized, 50, 150, 40)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.info("📝 Red grid helps with proportions, black lines show where to draw. Start light!")
        
        # Download button
        if len(result.shape) == 2:  # Grayscale
            result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        else:
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        result_pil = Image.fromarray(result_rgb)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        
        st.download_button(
            label="📥 Download Guide",
            data=buf.getvalue(),
            file_name=f"drawing_guide_{guide_type.replace(' ', '_').lower()}.png",
            mime="image/png"
        )
    
    # Drawing tips
    st.markdown("---")
    st.header("🎨 Drawing Tips for Beginners")
    
    tip_col1, tip_col2, tip_col3 = st.columns(3)
    
    with tip_col1:
        st.markdown("""
        **Getting Started:**
        1. Print or display the guide
        2. Start with light pencil strokes (2H or H)
        3. Draw basic shapes first
        4. Check proportions often
        """)
    
    with tip_col2:
        st.markdown("""
        **Building the Drawing:**
        1. Add medium tones (HB, B, 2B)
        2. Work from light to dark
        3. Blend with your finger or tissue
        4. Keep your pencil sharp
        """)
    
    with tip_col3:
        st.markdown("""
        **Finishing Touches:**
        1. Add darkest darks (4B-8B)
        2. Add highlights with eraser
        3. Refine edges and details
        4. Step back and assess
        """)

else:
    st.info("👆 Please upload a portrait image to begin learning!")
    
    # Show example workflow
    st.markdown("---")
    st.header("How to Use This App")
    
    st.markdown("""
    1. **Upload a Photo**: Choose a clear portrait photo (front-facing works best for beginners)
    2. **Select Guide Type**: 
       - Start with "Grid Overlay" to learn proportions
       - Use "Edge Detection" for outline practice
       - Try "Value Map" to understand shading
    3. **Adjust Settings**: Fine-tune the guide using the sidebar sliders
    4. **Download & Print**: Save the guide and print it or display it while drawing
    5. **Practice**: Draw along with the guide on your paper!
    
    **Pro Tip**: Start with the grid method to get proportions right, then move to edge detection for details!
    """)

# Footer
st.markdown("---")
st.markdown("*Happy Drawing! Practice makes perfect! 🎨*")

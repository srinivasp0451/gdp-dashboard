import streamlit as st
from PIL import Image, ImageFilter, ImageDraw, ImageOps, ImageEnhance
import numpy as np
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
    edge_strength = st.sidebar.slider("Edge Strength", 1, 10, 5)
    detail_level = st.sidebar.slider("Detail Level", 1, 5, 3)

if guide_type == "Grid Overlay":
    grid_size = st.sidebar.slider("Grid Cell Size", 20, 100, 40)
    grid_color = st.sidebar.selectbox("Grid Color", ["Gray", "Red", "Blue"])

if guide_type == "Value Map (Shading Guide)":
    tone_levels = st.sidebar.slider("Number of Tone Levels", 3, 10, 5)

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

def create_edge_detection(img, strength=5, detail=3):
    """Create edge detection guide using PIL"""
    # Convert to grayscale
    gray = img.convert('L')
    
    # Apply edge detection filters
    edges = gray.filter(ImageFilter.FIND_EDGES)
    
    # Enhance edges based on strength
    enhancer = ImageEnhance.Contrast(edges)
    edges = enhancer.enhance(strength * 0.5)
    
    # Additional edge enhancement
    edges = edges.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    # Apply threshold to make it more sketch-like
    threshold = 256 - (detail * 30)
    edges = edges.point(lambda x: 255 if x > threshold else 0)
    
    # Invert so edges are black on white
    edges = ImageOps.invert(edges)
    
    return edges

def create_simplified_sketch(img, strength=5, detail=3):
    """Create a simplified pencil sketch effect"""
    # Convert to grayscale
    gray = img.convert('L')
    
    # Invert the image
    inverted = ImageOps.invert(gray)
    
    # Apply blur to inverted image
    blur_radius = 10 + (detail * 2)
    blurred = inverted.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Invert the blurred image back
    inverted_blur = ImageOps.invert(blurred)
    
    # Create sketch effect by dividing
    gray_array = np.array(gray, dtype=np.float32)
    blur_array = np.array(inverted_blur, dtype=np.float32)
    
    # Avoid division by zero
    blur_array = np.where(blur_array == 0, 1, blur_array)
    
    # Create sketch
    sketch_array = (gray_array / blur_array) * 256.0
    sketch_array = np.clip(sketch_array, 0, 255).astype(np.uint8)
    
    sketch = Image.fromarray(sketch_array, mode='L')
    
    # Add edge detection for stronger outlines
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = edges.point(lambda x: 0 if x > (256 - detail * 20) else 255)
    
    # Combine sketch with edges
    sketch_array = np.array(sketch)
    edges_array = np.array(edges)
    combined = np.minimum(sketch_array, edges_array)
    
    result = Image.fromarray(combined, mode='L')
    
    # Enhance contrast based on strength
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(strength * 0.3)
    
    return result

def create_grid_overlay(img, grid_size, color="Gray"):
    """Create grid overlay for proportions"""
    output = img.copy().convert('RGB')
    draw = ImageDraw.Draw(output)
    
    w, h = output.size
    
    # Set color
    color_map = {
        "Gray": (100, 100, 100),
        "Red": (200, 50, 50),
        "Blue": (50, 50, 200)
    }
    line_color = color_map.get(color, (100, 100, 100))
    
    # Draw vertical lines
    for x in range(0, w, grid_size):
        draw.line([(x, 0), (x, h)], fill=line_color, width=1)
    
    # Draw horizontal lines
    for y in range(0, h, grid_size):
        draw.line([(0, y), (w, y)], fill=line_color, width=1)
    
    return output

def create_value_map(img, levels=5):
    """Create shading guide showing light and dark areas"""
    # Convert to grayscale
    gray = img.convert('L')
    
    # Apply blur to simplify values
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=5))
    
    # Create posterized version with specified tone levels
    img_array = np.array(blurred, dtype=np.float32)
    step = 255 / (levels - 1)
    posterized = np.round(img_array / step) * step
    posterized = np.clip(posterized, 0, 255).astype(np.uint8)
    
    result = Image.fromarray(posterized, mode='L')
    
    return result

def create_combined_guide(img, strength=5, detail=3, grid_size=40):
    """Create a combined guide with edges and grid"""
    # Start with edge detection
    edges = create_edge_detection(img, strength, detail)
    
    # Convert to RGB for grid overlay
    output = edges.convert('RGB')
    draw = ImageDraw.Draw(output)
    
    w, h = output.size
    
    # Add red grid
    for x in range(0, w, grid_size):
        draw.line([(x, 0), (x, h)], fill=(200, 100, 100), width=1)
    for y in range(0, h, grid_size):
        draw.line([(0, y), (w, y)], fill=(200, 100, 100), width=1)
    
    return output

def resize_image(img, max_size=800):
    """Resize image to fit display while maintaining aspect ratio"""
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img

# Main processing
if uploaded_file is not None:
    # Read the uploaded image
    img = Image.open(uploaded_file)
    
    # Convert RGBA to RGB if needed
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize for display
    img_resized = resize_image(img)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(img_resized, use_container_width=True)
    
    with col2:
        st.subheader(f"Drawing Guide: {guide_type}")
        
        # Generate the selected guide
        if guide_type == "Edge Detection (Contours)":
            result = create_edge_detection(img_resized, edge_strength, detail_level)
            st.image(result, use_container_width=True)
            st.info("📝 Trace these contour lines with a 2H or H pencil for your initial sketch.")
            
        elif guide_type == "Simplified Sketch":
            result = create_simplified_sketch(img_resized, edge_strength, detail_level)
            st.image(result, use_container_width=True)
            st.info("📝 Follow this sketch guide. Lighter areas = lighter pencil pressure, darker areas = more pressure.")
            
        elif guide_type == "Grid Overlay":
            result = create_grid_overlay(img_resized, grid_size, grid_color)
            st.image(result, use_container_width=True)
            st.info("📝 Draw the same grid on your paper. Use it to accurately place features and maintain proportions.")
            
        elif guide_type == "Value Map (Shading Guide)":
            result = create_value_map(img_resized, tone_levels)
            st.image(result, use_container_width=True)
            st.info("📝 This shows simplified shading zones. Darkest areas = 6B/8B, Medium = 2B/4B, Light = H/HB")
            
        elif guide_type == "Combined Guide":
            result = create_combined_guide(img_resized, edge_strength, detail_level, 40)
            st.image(result, use_container_width=True)
            st.info("📝 Red grid helps with proportions, black lines show where to draw. Start light!")
        
        # Download button
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        
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

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageFilter, ImageDraw, ImageOps, ImageEnhance
import numpy as np
import io

# Page configuration
st.set_page_config(
    page_title="Interactive Drawing Teacher",
    page_icon="✏️",
    layout="wide"
)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'processed_steps' not in st.session_state:
    st.session_state.processed_steps = {}

# Title
st.title("✏️ Interactive Portrait Drawing Teacher")
st.markdown("Upload a photo and learn to draw step-by-step!")

# Define pencil types
PENCILS = {
    '2H': {'color': '#C0C0C0', 'size': 2, 'description': 'Very light - Initial outlines'},
    'H': {'color': '#A0A0A0', 'size': 2, 'description': 'Light - Basic shapes'},
    'HB': {'color': '#808080', 'size': 3, 'description': 'Medium - General drawing'},
    'B': {'color': '#606060', 'size': 3, 'description': 'Medium-dark - Details'},
    '2B': {'color': '#505050', 'size': 4, 'description': 'Dark - Shading'},
    '4B': {'color': '#303030', 'size': 4, 'description': 'Very dark - Deep shadows'},
    '6B': {'color': '#202020', 'size': 5, 'description': 'Extra dark - Darkest areas'},
}

# Step configuration
STEPS = {
    1: {
        'name': 'Step 1: Basic Outline',
        'pencil': '2H',
        'description': 'Draw the basic outline and major shapes. Use very light strokes.',
        'edge_strength': 3,
        'detail_level': 1
    },
    2: {
        'name': 'Step 2: Add Features',
        'pencil': 'H',
        'description': 'Add eyes, nose, mouth and main facial features.',
        'edge_strength': 5,
        'detail_level': 2
    },
    3: {
        'name': 'Step 3: Add Shading',
        'pencil': '2B',
        'description': 'Add medium tones and basic shading to give depth.',
        'edge_strength': 6,
        'detail_level': 3
    },
    4: {
        'name': 'Step 4: Final Details',
        'pencil': '4B',
        'description': 'Add darkest shadows, fine details, and finishing touches.',
        'edge_strength': 8,
        'detail_level': 5
    }
}

def create_step_guide(img, step_num):
    """Create progressive guide for each step"""
    gray = img.convert('L')
    
    step_config = STEPS[step_num]
    strength = step_config['edge_strength']
    detail = step_config['detail_level']
    
    # Create base edges
    edges = gray.filter(ImageFilter.FIND_EDGES)
    enhancer = ImageEnhance.Contrast(edges)
    edges = enhancer.enhance(strength * 0.5)
    
    if step_num == 1:
        # Step 1: Only major outlines
        edges = edges.filter(ImageFilter.MaxFilter(5))
        threshold = 200
        edges = edges.point(lambda x: 255 if x > threshold else 0)
    
    elif step_num == 2:
        # Step 2: More details, facial features
        edges = edges.filter(ImageFilter.EDGE_ENHANCE_MORE)
        threshold = 180
        edges = edges.point(lambda x: 255 if x > threshold else 0)
    
    elif step_num == 3:
        # Step 3: Add shading guide
        edges = edges.filter(ImageFilter.EDGE_ENHANCE_MORE)
        threshold = 150
        edges = edges.point(lambda x: 255 if x > threshold else 0)
        
        # Add simple value map
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=3))
        img_array = np.array(blurred, dtype=np.float32)
        posterized = np.round(img_array / 50) * 50
        posterized = np.clip(posterized, 0, 255).astype(np.uint8)
        value_map = Image.fromarray(posterized, mode='L')
        
        # Blend edges with value map
        edges_array = np.array(edges)
        value_array = np.array(value_map)
        combined = np.minimum(edges_array, value_array)
        edges = Image.fromarray(combined, mode='L')
    
    else:  # Step 4
        # Step 4: Full detail
        edges = edges.filter(ImageFilter.EDGE_ENHANCE_MORE)
        edges = edges.filter(ImageFilter.EDGE_ENHANCE_MORE)
        threshold = 120
        edges = edges.point(lambda x: 255 if x > threshold else 0)
        
        # Add detailed value map
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=2))
        img_array = np.array(blurred, dtype=np.float32)
        posterized = np.round(img_array / 30) * 30
        posterized = np.clip(posterized, 0, 255).astype(np.uint8)
        value_map = Image.fromarray(posterized, mode='L')
        
        edges_array = np.array(edges)
        value_array = np.array(value_map)
        combined = np.minimum(edges_array, value_array)
        edges = Image.fromarray(combined, mode='L')
    
    # Invert so edges are black on white
    edges = ImageOps.invert(edges)
    
    return edges

def resize_image(img, max_size=600):
    """Resize image maintaining aspect ratio"""
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img

# Sidebar
with st.sidebar:
    st.header("📤 Upload Photo")
    uploaded_file = st.file_uploader("Choose a portrait", type=["jpg", "jpeg", "png"], key="uploader")
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        st.session_state.uploaded_image = resize_image(img)
        st.image(st.session_state.uploaded_image, caption="Your Photo", use_container_width=True)
    
    st.markdown("---")
    
    # Step navigation
    st.header("📚 Drawing Steps")
    
    cols = st.columns(4)
    for i in range(1, 5):
        with cols[i-1]:
            if st.button(f"Step {i}", key=f"step_{i}", 
                        type="primary" if st.session_state.current_step == i else "secondary",
                        use_container_width=True):
                st.session_state.current_step = i
    
    st.markdown("---")
    
    # Current step info
    current_step_config = STEPS[st.session_state.current_step]
    st.subheader(current_step_config['name'])
    st.info(current_step_config['description'])
    
    # Recommended pencil
    recommended_pencil = current_step_config['pencil']
    st.markdown(f"### ✏️ Recommended Pencil")
    st.markdown(f"**{recommended_pencil}** - {PENCILS[recommended_pencil]['description']}")
    
    st.markdown("---")
    
    # Guide settings
    st.header("👁️ Guide Settings")
    show_guide = st.checkbox("Show Guide Overlay", value=True)
    if show_guide:
        guide_opacity = st.slider("Guide Opacity", 0.0, 1.0, 0.5, 0.1)
    else:
        guide_opacity = 0.0

# Main area
if st.session_state.uploaded_image is None:
    st.info("👈 Please upload a portrait photo from the sidebar to begin!")
    st.markdown("### How to Use:")
    st.markdown("""
    1. **Upload** a portrait photo in the sidebar
    2. **Follow** the 4 progressive steps
    3. **Select** the recommended pencil for each step
    4. **Draw** directly on the canvas
    5. **Toggle** the guide overlay to check your progress
    """)
else:
    # Create guide for current step
    if st.session_state.current_step not in st.session_state.processed_steps:
        guide_img = create_step_guide(
            st.session_state.uploaded_image, 
            st.session_state.current_step
        )
        st.session_state.processed_steps[st.session_state.current_step] = guide_img
    else:
        guide_img = st.session_state.processed_steps[st.session_state.current_step]
    
    # Convert guide to RGB for canvas background
    guide_rgb = guide_img.convert('RGB')
    
    # Canvas size
    canvas_width = guide_rgb.size[0]
    canvas_height = guide_rgb.size[1]
    
    # Pencil selector
    st.markdown("### ✏️ Select Your Pencil")
    pencil_cols = st.columns(len(PENCILS))
    
    selected_pencil = st.session_state.get('selected_pencil', current_step_config['pencil'])
    
    for idx, (pencil_name, pencil_info) in enumerate(PENCILS.items()):
        with pencil_cols[idx]:
            is_recommended = (pencil_name == current_step_config['pencil'])
            button_label = f"{'⭐ ' if is_recommended else ''}{pencil_name}"
            
            if st.button(button_label, key=f"pencil_{pencil_name}", 
                        use_container_width=True,
                        type="primary" if selected_pencil == pencil_name else "secondary"):
                st.session_state.selected_pencil = pencil_name
                selected_pencil = pencil_name
    
    # Show selected pencil info
    st.markdown(f"**Selected: {selected_pencil}** - {PENCILS[selected_pencil]['description']}")
    
    st.markdown("---")
    
    # Tool selector
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        drawing_mode = st.radio("Tool:", ["Pencil ✏️", "Eraser 🧹"], horizontal=True)
    with col2:
        if st.button("🔄 Clear Canvas", use_container_width=True):
            st.rerun()
    
    # Prepare background
    if show_guide:
        # Create semi-transparent guide overlay
        guide_array = np.array(guide_rgb)
        white_bg = np.ones_like(guide_array) * 255
        blended = (guide_opacity * guide_array + (1 - guide_opacity) * white_bg).astype(np.uint8)
        background_image = Image.fromarray(blended)
    else:
        # White background
        background_image = Image.new('RGB', (canvas_width, canvas_height), 'white')
    
    # Drawing canvas
    st.markdown("### 🎨 Drawing Canvas")
    
    pencil_config = PENCILS[selected_pencil]
    stroke_width = pencil_config['size']
    stroke_color = pencil_config['color'] if drawing_mode == "Pencil ✏️" else "#FFFFFF"
    
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=background_image,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="freedraw" if drawing_mode == "Pencil ✏️" else "freedraw",
        key=f"canvas_step_{st.session_state.current_step}",
    )
    
    # Navigation buttons
    st.markdown("---")
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    
    with nav_col1:
        if st.session_state.current_step > 1:
            if st.button("⬅️ Previous Step", use_container_width=True):
                st.session_state.current_step -= 1
                st.rerun()
    
    with nav_col3:
        if st.session_state.current_step < 4:
            if st.button("Next Step ➡️", use_container_width=True):
                st.session_state.current_step += 1
                st.rerun()
        else:
            st.success("✅ Final Step Complete!")
    
    # Progress indicator
    progress = st.session_state.current_step / 4
    st.progress(progress, text=f"Progress: Step {st.session_state.current_step} of 4")
    
    # Tips for current step
    with st.expander("💡 Tips for This Step"):
        if st.session_state.current_step == 1:
            st.markdown("""
            - Use very light pressure
            - Focus on overall shapes, not details
            - Draw simple lines for head, shoulders, and main features
            - Don't worry about perfection!
            """)
        elif st.session_state.current_step == 2:
            st.markdown("""
            - Add eyes, nose, mouth positions
            - Draw eyebrows and ears
            - Still use light strokes
            - Check proportions against the guide
            """)
        elif st.session_state.current_step == 3:
            st.markdown("""
            - Start adding shadows on darker areas
            - Use medium pressure for shading
            - Blend with your finger for smooth transitions
            - Build up gradually
            """)
        else:
            st.markdown("""
            - Add the darkest darks
            - Refine all details
            - Add texture to hair
            - Clean up edges
            - Step back and look at the overall picture
            """)

# Footer
st.markdown("---")
st.markdown("*🎨 Practice daily to improve! Remember: Every artist was once a beginner!*")

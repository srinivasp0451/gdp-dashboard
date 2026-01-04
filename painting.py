import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageFilter, ImageDraw, ImageOps, ImageEnhance, ImageFont
import numpy as np
import io
import base64

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
if 'show_animation' not in st.session_state:
    st.session_state.show_animation = True

# Title
st.title("✏️ Interactive Portrait Drawing Teacher")
st.markdown("Upload a photo and learn to draw step-by-step with guided animations!")

# Define pencil/brush types
TOOLS = {
    'Pencil 2H': {'color': '#C0C0C0', 'size': 1, 'type': 'pencil'},
    'Pencil H': {'color': '#A0A0A0', 'size': 2, 'type': 'pencil'},
    'Pencil HB': {'color': '#808080', 'size': 3, 'type': 'pencil'},
    'Pencil B': {'color': '#606060', 'size': 3, 'type': 'pencil'},
    'Pencil 2B': {'color': '#505050', 'size': 4, 'type': 'pencil'},
    'Pencil 4B': {'color': '#303030', 'size': 5, 'type': 'pencil'},
    'Pencil 6B': {'color': '#202020', 'size': 6, 'type': 'pencil'},
    'Pencil 8B': {'color': '#101010', 'size': 7, 'type': 'pencil'},
    'Thin Brush': {'color': '#000000', 'size': 2, 'type': 'brush'},
    'Medium Brush': {'color': '#000000', 'size': 5, 'type': 'brush'},
    'Thick Brush': {'color': '#000000', 'size': 10, 'type': 'brush'},
    'Marker': {'color': '#000000', 'size': 15, 'type': 'marker'},
    'Highlighter': {'color': '#FFFF00', 'size': 20, 'type': 'highlighter'},
}

# Step configuration
STEPS = {
    1: {
        'name': 'Step 1: Basic Outline',
        'tool': 'Pencil 2H',
        'description': 'Draw the basic outline and major shapes. Use very light strokes.',
        'instruction': 'Trace the outer contour of the face and head shape',
        'edge_strength': 3,
        'detail_level': 1
    },
    2: {
        'name': 'Step 2: Add Features',
        'tool': 'Pencil H',
        'description': 'Add eyes, nose, mouth and main facial features.',
        'instruction': 'Draw the positions of eyes, nose, and mouth',
        'edge_strength': 5,
        'detail_level': 2
    },
    3: {
        'name': 'Step 3: Add Shading',
        'tool': 'Pencil 2B',
        'description': 'Add medium tones and basic shading to give depth.',
        'instruction': 'Shade the darker areas like under nose, eyes, and neck',
        'edge_strength': 6,
        'detail_level': 3
    },
    4: {
        'name': 'Step 4: Final Details',
        'tool': 'Pencil 4B',
        'description': 'Add darkest shadows, fine details, and finishing touches.',
        'instruction': 'Add the darkest shadows and refine all details',
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
    
    edges = gray.filter(ImageFilter.FIND_EDGES)
    enhancer = ImageEnhance.Contrast(edges)
    edges = enhancer.enhance(strength * 0.5)
    
    if step_num == 1:
        edges = edges.filter(ImageFilter.MaxFilter(5))
        threshold = 200
        edges = edges.point(lambda x: 255 if x > threshold else 0)
    elif step_num == 2:
        edges = edges.filter(ImageFilter.EDGE_ENHANCE_MORE)
        threshold = 180
        edges = edges.point(lambda x: 255 if x > threshold else 0)
    elif step_num == 3:
        edges = edges.filter(ImageFilter.EDGE_ENHANCE_MORE)
        threshold = 150
        edges = edges.point(lambda x: 255 if x > threshold else 0)
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=3))
        img_array = np.array(blurred, dtype=np.float32)
        posterized = np.round(img_array / 50) * 50
        posterized = np.clip(posterized, 0, 255).astype(np.uint8)
        value_map = Image.fromarray(posterized, mode='L')
        edges_array = np.array(edges)
        value_array = np.array(value_map)
        combined = np.minimum(edges_array, value_array)
        edges = Image.fromarray(combined, mode='L')
    else:
        edges = edges.filter(ImageFilter.EDGE_ENHANCE_MORE)
        edges = edges.filter(ImageFilter.EDGE_ENHANCE_MORE)
        threshold = 120
        edges = edges.point(lambda x: 255 if x > threshold else 0)
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=2))
        img_array = np.array(blurred, dtype=np.float32)
        posterized = np.round(img_array / 30) * 30
        posterized = np.clip(posterized, 0, 255).astype(np.uint8)
        value_map = Image.fromarray(posterized, mode='L')
        edges_array = np.array(edges)
        value_array = np.array(value_map)
        combined = np.minimum(edges_array, value_array)
        edges = Image.fromarray(combined, mode='L')
    
    edges = ImageOps.invert(edges)
    return edges

def resize_image(img, max_size=800):
    """Resize image maintaining aspect ratio"""
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img

def image_to_base64(img):
    """Convert PIL image to base64"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def create_drawing_canvas(background_img, width, height, tool_color, tool_size, show_guide, guide_opacity, show_animation, step_num):
    """Create advanced HTML canvas with MS Paint-like features"""
    bg_base64 = image_to_base64(background_img)
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <style>
            * {{
                box-sizing: border-box;
                -webkit-tap-highlight-color: transparent;
            }}
            body {{
                margin: 0;
                padding: 10px;
                background: #e0e0e0;
                font-family: Arial, sans-serif;
                overflow-x: hidden;
            }}
            .toolbar {{
                background: #f5f5f5;
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                position: sticky;
                top: 0;
                z-index: 100;
            }}
            button {{
                padding: 8px 12px;
                font-size: 14px;
                cursor: pointer;
                border: 2px solid #333;
                border-radius: 5px;
                background: white;
                font-weight: bold;
                transition: all 0.2s;
                flex: 1 1 auto;
                min-width: 80px;
            }}
            button:active {{
                transform: scale(0.95);
            }}
            button.active {{
                background: #4CAF50;
                color: white;
            }}
            button.danger {{
                background: #f44336;
                color: white;
            }}
            .tool-group {{
                display: flex;
                gap: 5px;
                flex-wrap: wrap;
                width: 100%;
            }}
            #canvasContainer {{
                position: relative;
                border: 3px solid #333;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                background: white;
                margin: 0 auto;
                max-width: 100%;
                overflow: auto;
                -webkit-overflow-scrolling: touch;
            }}
            canvas {{
                display: block;
                touch-action: none;
            }}
            #guideCanvas {{
                position: absolute;
                top: 0;
                left: 0;
                opacity: {guide_opacity if show_guide else 0};
                pointer-events: none;
            }}
            #animationCanvas {{
                position: absolute;
                top: 0;
                left: 0;
                pointer-events: none;
                z-index: 10;
            }}
            #drawingCanvas {{
                cursor: crosshair;
            }}
            .color-picker {{
                width: 50px;
                height: 40px;
                border: 2px solid #333;
                border-radius: 5px;
                cursor: pointer;
            }}
            input[type="range"] {{
                width: 100px;
            }}
            .size-display {{
                padding: 8px;
                background: white;
                border: 2px solid #333;
                border-radius: 5px;
                min-width: 60px;
                text-align: center;
            }}
            #textInput {{
                padding: 8px;
                border: 2px solid #333;
                border-radius: 5px;
                display: none;
            }}
            .instruction {{
                background: #fffacd;
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 10px;
                border: 2px solid #ffd700;
                font-weight: bold;
                text-align: center;
            }}
            @media (max-width: 768px) {{
                button {{
                    font-size: 12px;
                    padding: 6px 10px;
                    min-width: 70px;
                }}
                .toolbar {{
                    padding: 5px;
                    gap: 5px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="instruction" id="instruction">
            📝 {STEPS[step_num]['instruction']}
        </div>
        
        <div class="toolbar">
            <div class="tool-group">
                <button onclick="setTool('pencil')" id="pencilBtn" class="active">✏️ Pencil</button>
                <button onclick="setTool('brush')" id="brushBtn">🖌️ Brush</button>
                <button onclick="setTool('eraser')" id="eraserBtn">🧹 Eraser</button>
                <button onclick="setTool('line')" id="lineBtn">📏 Line</button>
                <button onclick="setTool('rectangle')" id="rectangleBtn">⬜ Rectangle</button>
                <button onclick="setTool('circle')" id="circleBtn">⭕ Circle</button>
                <button onclick="setTool('text')" id="textBtn">📝 Text</button>
            </div>
            
            <div class="tool-group">
                <button onclick="undo()">↶ Undo</button>
                <button onclick="redo()">↷ Redo</button>
                <button onclick="clearCanvas()" class="danger">🗑️ Clear</button>
            </div>
            
            <div class="tool-group">
                <input type="color" id="colorPicker" class="color-picker" value="{tool_color}">
                <input type="range" id="sizeSlider" min="1" max="50" value="{tool_size}">
                <span class="size-display" id="sizeDisplay">{tool_size}px</span>
            </div>
            
            <div class="tool-group">
                <button onclick="toggleGuide()" id="toggleBtn">👁️ Guide</button>
                <button onclick="toggleAnimation()" id="animBtn">🎬 Animation</button>
                <button onclick="downloadDrawing()">📥 Save</button>
            </div>
        </div>
        
        <input type="text" id="textInput" placeholder="Type text and click on canvas">
        
        <div id="canvasContainer">
            <canvas id="guideCanvas" width="{width}" height="{height}"></canvas>
            <canvas id="animationCanvas" width="{width}" height="{height}"></canvas>
            <canvas id="drawingCanvas" width="{width}" height="{height}"></canvas>
        </div>

        <script>
            const guideCanvas = document.getElementById('guideCanvas');
            const guideCtx = guideCanvas.getContext('2d');
            const animCanvas = document.getElementById('animationCanvas');
            const animCtx = animCanvas.getContext('2d');
            const drawingCanvas = document.getElementById('drawingCanvas');
            const ctx = drawingCanvas.getContext('2d');
            
            let currentTool = 'pencil';
            let isDrawing = false;
            let lastX = 0;
            let lastY = 0;
            let startX = 0;
            let startY = 0;
            let currentColor = '{tool_color}';
            let currentSize = {tool_size};
            let guideVisible = {str(show_guide).lower()};
            let animationVisible = {str(show_animation).lower()};
            let history = [];
            let historyStep = -1;
            
            // Load background
            const img = new Image();
            img.onload = function() {{
                guideCtx.drawImage(img, 0, 0, {width}, {height});
                saveState();
                if (animationVisible) startAnimation();
            }};
            img.src = 'data:image/png;base64,{bg_base64}';
            
            // Tool settings
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            
            // Color picker
            document.getElementById('colorPicker').addEventListener('change', (e) => {{
                currentColor = e.target.value;
            }});
            
            // Size slider
            document.getElementById('sizeSlider').addEventListener('input', (e) => {{
                currentSize = parseInt(e.target.value);
                document.getElementById('sizeDisplay').textContent = currentSize + 'px';
            }});
            
            function setTool(tool) {{
                currentTool = tool;
                document.querySelectorAll('.toolbar button').forEach(btn => btn.classList.remove('active'));
                document.getElementById(tool + 'Btn').classList.add('active');
                
                if (tool === 'text') {{
                    document.getElementById('textInput').style.display = 'block';
                }} else {{
                    document.getElementById('textInput').style.display = 'none';
                }}
            }}
            
            function saveState() {{
                historyStep++;
                if (historyStep < history.length) {{
                    history.length = historyStep;
                }}
                history.push(drawingCanvas.toDataURL());
            }}
            
            function undo() {{
                if (historyStep > 0) {{
                    historyStep--;
                    const img = new Image();
                    img.onload = function() {{
                        ctx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
                        ctx.drawImage(img, 0, 0);
                    }};
                    img.src = history[historyStep];
                }}
            }}
            
            function redo() {{
                if (historyStep < history.length - 1) {{
                    historyStep++;
                    const img = new Image();
                    img.onload = function() {{
                        ctx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
                        ctx.drawImage(img, 0, 0);
                    }};
                    img.src = history[historyStep];
                }}
            }}
            
            // Mouse events
            drawingCanvas.addEventListener('mousedown', startDrawing);
            drawingCanvas.addEventListener('mousemove', draw);
            drawingCanvas.addEventListener('mouseup', stopDrawing);
            drawingCanvas.addEventListener('mouseout', stopDrawing);
            
            // Touch events
            drawingCanvas.addEventListener('touchstart', handleTouch);
            drawingCanvas.addEventListener('touchmove', handleTouch);
            drawingCanvas.addEventListener('touchend', stopDrawing);
            
            function getCoordinates(e) {{
                const rect = drawingCanvas.getBoundingClientRect();
                const scaleX = drawingCanvas.width / rect.width;
                const scaleY = drawingCanvas.height / rect.height;
                
                if (e.touches) {{
                    return {{
                        x: (e.touches[0].clientX - rect.left) * scaleX,
                        y: (e.touches[0].clientY - rect.top) * scaleY
                    }};
                }}
                return {{
                    x: (e.clientX - rect.left) * scaleX,
                    y: (e.clientY - rect.top) * scaleY
                }};
            }}
            
            function startDrawing(e) {{
                e.preventDefault();
                isDrawing = true;
                const coords = getCoordinates(e);
                lastX = startX = coords.x;
                lastY = startY = coords.y;
                
                if (currentTool === 'text') {{
                    const text = document.getElementById('textInput').value;
                    if (text) {{
                        ctx.font = currentSize * 2 + 'px Arial';
                        ctx.fillStyle = currentColor;
                        ctx.fillText(text, coords.x, coords.y);
                        saveState();
                    }}
                    isDrawing = false;
                }}
            }}
            
            function draw(e) {{
                if (!isDrawing) return;
                e.preventDefault();
                
                const coords = getCoordinates(e);
                const x = coords.x;
                const y = coords.y;
                
                ctx.strokeStyle = currentColor;
                ctx.fillStyle = currentColor;
                ctx.lineWidth = currentSize;
                
                if (currentTool === 'pencil' || currentTool === 'brush') {{
                    ctx.beginPath();
                    ctx.moveTo(lastX, lastY);
                    ctx.lineTo(x, y);
                    ctx.stroke();
                    lastX = x;
                    lastY = y;
                }} else if (currentTool === 'eraser') {{
                    ctx.clearRect(x - currentSize/2, y - currentSize/2, currentSize, currentSize);
                }}
            }}
            
            function stopDrawing(e) {{
                if (!isDrawing) return;
                
                if (currentTool === 'line' || currentTool === 'rectangle' || currentTool === 'circle') {{
                    const coords = getCoordinates(e);
                    const x = coords.x;
                    const y = coords.y;
                    
                    ctx.strokeStyle = currentColor;
                    ctx.fillStyle = currentColor;
                    ctx.lineWidth = currentSize;
                    
                    if (currentTool === 'line') {{
                        ctx.beginPath();
                        ctx.moveTo(startX, startY);
                        ctx.lineTo(x, y);
                        ctx.stroke();
                    }} else if (currentTool === 'rectangle') {{
                        ctx.strokeRect(startX, startY, x - startX, y - startY);
                    }} else if (currentTool === 'circle') {{
                        const radius = Math.sqrt(Math.pow(x - startX, 2) + Math.pow(y - startY, 2));
                        ctx.beginPath();
                        ctx.arc(startX, startY, radius, 0, 2 * Math.PI);
                        ctx.stroke();
                    }}
                }}
                
                isDrawing = false;
                saveState();
            }}
            
            function handleTouch(e) {{
                e.preventDefault();
                if (e.type === 'touchstart') {{
                    startDrawing(e);
                }} else if (e.type === 'touchmove') {{
                    draw(e);
                }}
            }}
            
            function clearCanvas() {{
                if (confirm('Clear your drawing?')) {{
                    ctx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
                    saveState();
                }}
            }}
            
            function toggleGuide() {{
                guideVisible = !guideVisible;
                guideCanvas.style.opacity = guideVisible ? '{guide_opacity}' : '0';
                document.getElementById('toggleBtn').textContent = guideVisible ? '👁️ Hide Guide' : '👁️ Show Guide';
            }}
            
            function toggleAnimation() {{
                animationVisible = !animationVisible;
                if (animationVisible) {{
                    startAnimation();
                }} else {{
                    animCtx.clearRect(0, 0, animCanvas.width, animCanvas.height);
                }}
                document.getElementById('animBtn').textContent = animationVisible ? '⏸️ Pause' : '▶️ Play';
            }}
            
            function startAnimation() {{
                // Simple animation showing drawing path
                let progress = 0;
                const animate = () => {{
                    if (!animationVisible) return;
                    
                    animCtx.clearRect(0, 0, animCanvas.width, animCanvas.height);
                    animCtx.strokeStyle = '#FF0000';
                    animCtx.lineWidth = 3;
                    animCtx.setLineDash([5, 5]);
                    
                    // Draw animated guide (simplified)
                    progress = (progress + 0.02) % 1;
                    animCtx.globalAlpha = Math.sin(progress * Math.PI);
                    
                    setTimeout(() => requestAnimationFrame(animate), 50);
                }};
                animate();
            }}
            
            function downloadDrawing() {{
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = {width};
                tempCanvas.height = {height};
                const tempCtx = tempCanvas.getContext('2d');
                
                tempCtx.fillStyle = 'white';
                tempCtx.fillRect(0, 0, {width}, {height});
                tempCtx.drawImage(drawingCanvas, 0, 0);
                
                const link = document.createElement('a');
                link.download = 'my_drawing_step_{step_num}.png';
                link.href = tempCanvas.toDataURL();
                link.click();
            }}
        </script>
    </body>
    </html>
    """
    
    return html_code

# Sidebar
with st.sidebar:
    st.header("📤 Upload Photo")
    uploaded_file = st.file_uploader("Choose a portrait", type=["jpg", "jpeg", "png", "bmp", "webp"], key="uploader")
    
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
    
    st.header("📚 Learning Steps")
    for i in range(1, 5):
        status = '✅' if i < st.session_state.current_step else ('👉' if i == st.session_state.current_step else '⭕')
        if st.button(f"{status} {STEPS[i]['name']}", 
                    key=f"step_{i}", 
                    use_container_width=True,
                    type="primary" if st.session_state.current_step == i else "secondary"):
            st.session_state.current_step = i
            st.rerun()
    
    st.markdown("---")
    
    current_step_config = STEPS[st.session_state.current_step]
    st.subheader(f"Current: Step {st.session_state.current_step}")
    st.info(current_step_config['description'])
    
    st.markdown(f"### ✏️ Recommended Tool")
    st.markdown(f"**{current_step_config['tool']}**")
    
    st.markdown("---")
    
    st.header("⚙️ Settings")
    show_guide = st.checkbox("Show Guide Overlay", value=True)
    if show_guide:
        guide_opacity = st.slider("Guide Opacity", 0.0, 1.0, 0.4, 0.1)
    else:
        guide_opacity = 0.0
    
    show_animation = st.checkbox("Show Animated Guide", value=st.session_state.show_animation)
    st.session_state.show_animation = show_animation

# Main area
if st.session_state.uploaded_image is None:
    st.info("👈 Please upload a portrait photo from the sidebar!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎨 Features:")
        st.markdown("""
        - **Full MS Paint Tools**: Pencil, Brush, Eraser, Shapes, Text
        - **8+ Pencil Types**: From 2H (lightest) to 8B (darkest)
        - **Undo/Redo**: Fix mistakes easily
        - **Mobile Friendly**: Touch support with scrolling
        - **Color Picker**: Choose any color
        - **Size Control**: Adjust brush/pencil size
        - **Download**: Save your artwork
        """)
    
    with col2:
        st.markdown("### 📖 How to Use:")
        st.markdown("""
        1. Upload a portrait photo
        2. Follow 4 progressive steps
        3. Use recommended tools
        4. Draw with guided animations
        5. Toggle guide visibility
        6. Use shapes and text tools
        7. Undo mistakes anytime
        8. Download your masterpiece!
        """)
else:
    if st.session_state.current_step not in st.session_state.processed_steps:
        guide_img = create_step_guide(st.session_state.uploaded_image, st.session_state.current_step)
        st.session_state.processed_steps[st.session_state.current_step] = guide_img
    else:
        guide_img = st.session_state.processed_steps[st.session_state.current_step]
    
    guide_rgb = guide_img.convert('RGB')
    canvas_width = guide_rgb.size[0]
    canvas_height = guide_rgb.size[1]
    
    if show_guide:
        guide_array = np.array(guide_rgb)
        white_bg = np.ones_like(guide_array) * 255
        blended = (guide_opacity * guide_array + (1 - guide_opacity) * white_bg).astype(np.uint8)
        background_image = Image.fromarray(blended)
    else:
        background_image = Image.new('RGB', (canvas_width, canvas_height), 'white')
    
    recommended_tool = STEPS[st.session_state.current_step]['tool']
    tool_config = TOOLS[recommended_tool]
    
    canvas_html = create_drawing_canvas(
        background_image,
        canvas_width,
        canvas_height,
        tool_config['color'],
        tool_config['size'],
        show_guide,
        guide_opacity,
        show_animation,
        st.session_state.current_step
    )
    
    # Responsive height calculation
    display_height = min(canvas_height + 250, 1200)
    components.html(canvas_html, height=display_height, scrolling=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.session_state.current_step > 1:
            if st.button("⬅️ Previous Step", use_container_width=True):
                st.session_state.current_step -= 1
                st.rerun()
    
    with col2:
        progress = st.session_state.current_step / 4
        st.progress(progress, text=f"Step {st.session_state.current_step} of 4")
    
    with col3:
        if st.session_state.current_step < 4:
            if st.button("Next Step ➡️", use_container_width=True):
                st.session_state.current_step += 1
                st.rerun()
        else:
            st.success("🎉 Complete!")

st.markdown("---")
st.markdown("*🎨 Keep practicing! Every stroke makes you better!*")

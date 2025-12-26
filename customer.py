import streamlit as st
import json
import os
from datetime import datetime, timedelta
from PIL import Image
import base64
from io import BytesIO

# Set page config
st.set_page_config(page_title="MediFind - Customer", page_icon="💊", layout="wide")

# Shared data file - using absolute path to ensure both apps use same file
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'shared_requests.json')

# Initialize or load data
def load_data():
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r') as f:
                content = f.read()
                if content.strip():
                    return json.loads(content)
                return {'requests': []}
        return {'requests': []}
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {'requests': []}

def save_data(data):
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False

def cleanup_old_requests(data):
    """Remove requests older than 24 hours"""
    current_time = datetime.now()
    data['requests'] = [
        req for req in data['requests']
        if (current_time - datetime.strptime(req['timestamp'], '%Y-%m-%d %H:%M:%S')) < timedelta(hours=24)
    ]
    return data

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        padding: 1.5rem;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        border-radius: 10px;
        color: #155724;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .info-box {
        padding: 1.5rem;
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 5px solid #17a2b8;
        border-radius: 10px;
        color: #0c5460;
        margin: 1rem 0;
    }
    .response-card {
        padding: 1.5rem;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">💊 MediFind - Find Your Medicine</h1>', unsafe_allow_html=True)

# Load and cleanup data
data = load_data()
data = cleanup_old_requests(data)
save_data(data)

# Show debug info
with st.expander("🔧 Debug Info (Click to expand)"):
    st.code(f"Data file location: {DATA_FILE}")
    st.code(f"File exists: {os.path.exists(DATA_FILE)}")
    st.code(f"Total requests in file: {len(data['requests'])}")
    if st.button("View Raw Data"):
        st.json(data)

# Initialize session state
if 'current_request_id' not in st.session_state:
    st.session_state.current_request_id = None

# Main form
st.markdown("### 📋 Request Medicine from Nearby Shops")

col1, col2 = st.columns([3, 2])

with col1:
    medicine_name = st.text_input("💊 Medicine Name *", placeholder="Enter medicine name", key="med_name")
    medicine_mg = st.text_input("⚖️ Dosage (Optional)", placeholder="e.g., 500mg, 10ml", key="med_mg")
    uploaded_file = st.file_uploader("📸 Upload Medicine Picture", 
                                    type=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

with col2:
    st.markdown("""
    <div class="info-box">
    <h4>📝 How it works:</h4>
    <ol>
    <li>Enter medicine name</li>
    <li>Add dosage (optional)</li>
    <li>Upload picture (optional)</li>
    <li>Click 'Request Medicine'</li>
    <li>Wait for shop responses</li>
    </ol>
    <p><strong>Note:</strong> Requests auto-delete after 24 hours</p>
    </div>
    """, unsafe_allow_html=True)

# Request button
if st.button("🔍 Request Medicine from Nearby Shops", type="primary", use_container_width=True):
    if not medicine_name:
        st.error("⚠️ Please enter the medicine name!")
    else:
        request_id = f"REQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert image to base64 if uploaded
        image_data = None
        if uploaded_file is not None:
            image_data = image_to_base64(image)
        
        # Create request
        new_request = {
            'request_id': request_id,
            'medicine_name': medicine_name,
            'medicine_mg': medicine_mg if medicine_mg else 'Not specified',
            'image_data': image_data,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'pending',
            'responses': []
        }
        
        # Add to data
        data['requests'].append(new_request)
        if save_data(data):
            st.session_state.current_request_id = request_id
            st.markdown(f"""
            <div class="success-box">
            ✅ <strong>Request submitted successfully!</strong><br>
            Request ID: <strong>{request_id}</strong><br>
            Medicine: <strong>{medicine_name}</strong><br>
            Time: <strong>{new_request['timestamp']}</strong><br>
            Data file: <code>{DATA_FILE}</code>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.error("Failed to save request. Please check file permissions.")

# Display responses section
st.markdown("---")
st.markdown("### 📬 Your Active Requests & Responses")

# Reload data to check for new responses
data = load_data()

# Find user's requests (for demo, showing all requests)
user_requests = [req for req in data['requests']]

if not user_requests:
    st.info("📭 No active requests. Submit a request above to get started!")
else:
    for request in sorted(user_requests, key=lambda x: x['timestamp'], reverse=True):
        response_count = len(request.get('responses', []))
        
        with st.expander(f"{'✅' if response_count > 0 else '⏳'} {request['medicine_name']} - {request['request_id']}", 
                        expanded=(response_count > 0)):
            
            col_a, col_b = st.columns([2, 1])
            
            with col_a:
                st.markdown(f"**Medicine:** {request['medicine_name']}")
                st.markdown(f"**Dosage:** {request['medicine_mg']}")
                st.markdown(f"**Submitted:** {request['timestamp']}")
                st.markdown(f"**Status:** {'✅ Received Responses' if response_count > 0 else '⏳ Waiting...'}")
                
                # Show image if available
                if request.get('image_data'):
                    img_bytes = base64.b64decode(request['image_data'])
                    img = Image.open(BytesIO(img_bytes))
                    st.image(img, caption="Medicine Image", width=250)
            
            with col_b:
                if response_count > 0:
                    st.success(f"🎉 {response_count} Response(s)")
                else:
                    st.warning("Waiting for shops...")
            
            # Display responses
            if request.get('responses'):
                st.markdown("---")
                st.markdown("#### 🏪 Shop Responses:")
                
                for idx, response in enumerate(request['responses'], 1):
                    st.markdown(f"""
                    <div class="response-card">
                    <h4>Response #{idx}</h4>
                    <p><strong>🏪 Shop:</strong> {response['shop_name']}</p>
                    <p><strong>📍 Address:</strong> {response['address']}</p>
                    <p><strong>📞 Contact:</strong> {response.get('contact', 'Not provided')}</p>
                    <p><strong>⏰ Responded:</strong> {response['response_time']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if response.get('location'):
                        st.markdown(f"[🗺️ **View Location on Google Maps**]({response['location']})")
                    
                    st.markdown("")

# Auto-refresh section
st.markdown("---")
col_x, col_y = st.columns([3, 1])
with col_x:
    st.info("💡 **Tip:** Enable auto-refresh to get real-time updates when shops respond!")
with col_y:
    if st.button("🔄 Refresh Now"):
        st.rerun()

if st.checkbox("🔄 Auto-refresh (every 10 seconds)"):
    import time
    time.sleep(10)
    st.rerun()

# Footer
st.markdown("---")
st.caption("💊 MediFind - Connecting patients with nearby medical shops | Data auto-cleans after 24 hours")

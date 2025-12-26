import streamlit as st
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO
import json
import hashlib

# Set page config
st.set_page_config(page_title="MediFind - Customer", page_icon="💊", layout="wide")

# Shared storage key (use URL parameter to sync between apps)
STORAGE_KEY = 'medifind_requests'

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def get_storage_data():
    """Get data from URL params"""
    params = st.query_params
    if 'data' in params:
        try:
            return json.loads(params['data'])
        except:
            return {'requests': []}
    return {'requests': []}

def save_storage_data(data):
    """Save data to URL params"""
    st.query_params['data'] = json.dumps(data)

# Initialize from URL or create new
if 'initialized' not in st.session_state:
    url_data = get_storage_data()
    if url_data['requests']:
        st.session_state.requests = url_data['requests']
    else:
        st.session_state.requests = []
    st.session_state.initialized = True
    st.session_state.current_request_id = None

# Sync with URL params on each load
url_data = get_storage_data()
if url_data['requests']:
    # Update responses from vendor
    for url_req in url_data['requests']:
        for session_req in st.session_state.requests:
            if session_req['request_id'] == url_req['request_id']:
                session_req['responses'] = url_req.get('responses', [])

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
    }
    .share-url-box {
        background: #f8f9fa;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        word-break: break-all;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">💊 MediFind - Find Your Medicine</h1>', unsafe_allow_html=True)

# Vendor URL Configuration
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("### 🔗 Share URL with Vendor")

# Generate shareable URL
current_url = st.query_params.get('data', '')
if current_url:
    vendor_url = f"Open vendor app and import this URL"
    st.sidebar.code(f"{st.get_option('browser.serverAddress')}:{st.get_option('browser.serverPort')}")
else:
    st.sidebar.info("Submit a request first, then share the URL with vendor")

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
    <li>Share URL with vendor</li>
    <li>Get real-time responses!</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# Request button
if st.button("🔍 Request Medicine from Nearby Shops", type="primary", use_container_width=True):
    if not medicine_name:
        st.error("⚠️ Please enter the medicine name!")
    else:
        request_id = f"REQ_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
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
        
        # Add to session
        st.session_state.requests.append(new_request)
        st.session_state.current_request_id = request_id
        
        # Save to URL params
        save_storage_data({'requests': st.session_state.requests})
        
        st.markdown(f"""
        <div class="success-box">
        ✅ <strong>Request submitted successfully!</strong><br>
        Request ID: <strong>{request_id}</strong><br>
        Medicine: <strong>{medicine_name}</strong><br>
        Time: <strong>{new_request['timestamp']}</strong>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
        st.rerun()

# Share URL section
if st.session_state.requests:
    st.markdown("---")
    st.markdown("### 🔗 Share This URL with Vendor")
    
    # Create shareable link
    current_data = json.dumps({'requests': st.session_state.requests})
    share_url = f"Copy the current page URL and share with vendor app"
    
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.info("📋 **Copy current page URL from browser address bar and paste in vendor app**")
    with col_b:
        if st.button("📋 How to Share"):
            st.info("""
            1. Copy URL from browser
            2. Send to vendor
            3. Vendor opens vendor app
            4. Vendor pastes URL
            5. Real-time sync!
            """)

# Display requests section
st.markdown("---")
st.markdown("### 📬 Your Active Requests & Responses")

if not st.session_state.requests:
    st.info("📭 No active requests. Submit a request above to get started!")
else:
    for request in sorted(st.session_state.requests, key=lambda x: x['timestamp'], reverse=True):
        response_count = len(request.get('responses', []))
        
        with st.expander(
            f"{'✅' if response_count > 0 else '⏳'} {request['medicine_name']} - {request['request_id']}", 
            expanded=(response_count > 0)
        ):
            col_a, col_b = st.columns([2, 1])
            
            with col_a:
                st.markdown(f"**Medicine:** {request['medicine_name']}")
                st.markdown(f"**Dosage:** {request['medicine_mg']}")
                st.markdown(f"**Submitted:** {request['timestamp']}")
                st.markdown(f"**Status:** {'✅ Received Responses' if response_count > 0 else '⏳ Waiting...'}")
                
                # Show image if available
                if request.get('image_data'):
                    try:
                        img_bytes = base64.b64decode(request['image_data'])
                        img = Image.open(BytesIO(img_bytes))
                        st.image(img, caption="Medicine Image", width=250)
                    except:
                        st.warning("Unable to display image")
            
            with col_b:
                if response_count > 0:
                    st.success(f"🎉 {response_count} Response(s)")
                else:
                    st.warning("Waiting for vendor...")
            
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

# Clear data option
st.markdown("---")
if st.button("🗑️ Clear All Requests"):
    st.session_state.requests = []
    st.session_state.current_request_id = None
    st.query_params.clear()
    st.success("✅ All requests cleared!")
    st.rerun()

# Auto-refresh
st.markdown("---")
col_ref1, col_ref2 = st.columns([3, 1])
with col_ref1:
    st.info("💡 **Tip:** Keep auto-refresh ON to receive vendor responses in real-time!")
with col_ref2:
    if st.button("🔄 Refresh"):
        st.rerun()

auto_refresh = st.checkbox("🔄 Auto-refresh (every 5 seconds)", value=True)
if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()

# Footer
st.markdown("---")
st.caption("💊 MediFind Customer App | Real-time Communication via URL Sharing")

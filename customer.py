import streamlit as st
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO
import json

# Set page config
st.set_page_config(page_title="MediFind - Customer", page_icon="💊", layout="wide")

# Initialize session state for in-memory storage
if 'requests' not in st.session_state:
    st.session_state.requests = []
if 'current_request_id' not in st.session_state:
    st.session_state.current_request_id = None

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
    .api-box {
        background: #f8f9fa;
        border: 2px dashed #6c757d;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">💊 MediFind - Find Your Medicine</h1>', unsafe_allow_html=True)

# Vendor API URL input
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("### Vendor App URL")
vendor_url = st.sidebar.text_input(
    "Enter Vendor App URL",
    placeholder="https://your-vendor-app.streamlit.app",
    help="The URL where your vendor app is hosted"
)
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Get the vendor URL from your hosting platform after deploying vendor_app.py")

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
    <li>Configure vendor URL in sidebar</li>
    <li>Enter medicine name</li>
    <li>Add dosage (optional)</li>
    <li>Upload picture (optional)</li>
    <li>Click 'Request Medicine'</li>
    <li>Wait for shop responses</li>
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
        
        # Store in session state
        st.session_state.requests.append(new_request)
        st.session_state.current_request_id = request_id
        
        st.markdown(f"""
        <div class="success-box">
        ✅ <strong>Request created successfully!</strong><br>
        Request ID: <strong>{request_id}</strong><br>
        Medicine: <strong>{medicine_name}</strong><br>
        Time: <strong>{new_request['timestamp']}</strong>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
        
        # Show API payload for vendor
        st.markdown("### 📡 Request Data (Share this with vendor)")
        st.json(new_request)

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
            
            # Manual response input section
            st.markdown("---")
            st.markdown("#### 📥 Add Vendor Response (Manual)")
            st.info("💡 Vendor will provide response data. Copy and paste it here.")
            
            with st.form(key=f"response_form_{request['request_id']}"):
                response_json = st.text_area(
                    "Paste vendor response JSON here",
                    height=150,
                    placeholder='{"shop_name": "ABC Pharmacy", "address": "123 Main St", ...}'
                )
                
                if st.form_submit_button("➕ Add Response"):
                    try:
                        response_data = json.loads(response_json)
                        
                        # Find and update request
                        for req in st.session_state.requests:
                            if req['request_id'] == request['request_id']:
                                req['responses'].append(response_data)
                                req['status'] = 'responded'
                                break
                        
                        st.success("✅ Response added successfully!")
                        st.rerun()
                    except json.JSONDecodeError:
                        st.error("❌ Invalid JSON format. Please check the response data.")

# API Information Section
st.markdown("---")
st.markdown("### 📡 API Integration Guide")

with st.expander("🔌 How to Connect with Vendor App"):
    st.markdown("""
    #### Method 1: Share Request Data
    1. Submit a request above
    2. Copy the JSON data shown
    3. Send to vendor via any method (email, chat, etc.)
    4. Vendor processes and sends back response JSON
    5. Paste vendor's response in "Add Vendor Response" section
    
    #### Method 2: Direct URL Communication (Advanced)
    1. Configure vendor URL in sidebar
    2. Both apps can read from each other's query parameters
    3. Use the data export/import features below
    
    #### Request Data Format:
    ```json
    {
        "request_id": "REQ_20241226_143022_123456",
        "medicine_name": "Paracetamol",
        "medicine_mg": "500mg",
        "image_data": "base64_encoded_string",
        "timestamp": "2024-12-26 14:30:22",
        "status": "pending",
        "responses": []
    }
    ```
    
    #### Response Data Format:
    ```json
    {
        "shop_name": "ABC Pharmacy",
        "address": "123 Main Street, City",
        "contact": "+1234567890",
        "location": "https://maps.google.com/?q=lat,lng",
        "response_time": "2024-12-26 14:35:00"
    }
    ```
    """)

# Data Export/Import
st.markdown("---")
col_exp, col_imp = st.columns(2)

with col_exp:
    st.markdown("#### 📤 Export All Requests")
    if st.button("Export Requests as JSON"):
        export_data = {
            'requests': st.session_state.requests,
            'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        st.download_button(
            label="⬇️ Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"requests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

with col_imp:
    st.markdown("#### 📥 Import Requests")
    uploaded_json = st.file_uploader("Upload JSON file", type=['json'], key="import_file")
    if uploaded_json:
        try:
            import_data = json.load(uploaded_json)
            if st.button("Import Data"):
                st.session_state.requests = import_data.get('requests', [])
                st.success("✅ Data imported successfully!")
                st.rerun()
        except:
            st.error("❌ Invalid JSON file")

# Clear data option
st.markdown("---")
if st.button("🗑️ Clear All Requests", type="secondary"):
    st.session_state.requests = []
    st.session_state.current_request_id = None
    st.success("✅ All requests cleared!")
    st.rerun()

# Auto-refresh
st.markdown("---")
col_ref1, col_ref2 = st.columns([3, 1])
with col_ref1:
    st.info("💡 **Tip:** Enable auto-refresh to check for new responses automatically!")
with col_ref2:
    if st.button("🔄 Refresh Now"):
        st.rerun()

if st.checkbox("🔄 Auto-refresh (every 10 seconds)"):
    import time
    time.sleep(10)
    st.rerun()

# Footer
st.markdown("---")
st.caption("💊 MediFind Customer App | API-Based Communication | Session Storage")

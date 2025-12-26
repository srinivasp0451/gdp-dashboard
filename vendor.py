import streamlit as st
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO
import json

# Set page config
st.set_page_config(page_title="MediFind - Vendor", page_icon="🏪", layout="wide")

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

def count_pending_requests(requests):
    """Count requests without responses"""
    return sum(1 for req in requests if not req.get('responses'))

# Initialize from URL
if 'initialized' not in st.session_state:
    url_data = get_storage_data()
    st.session_state.requests = url_data.get('requests', [])
    st.session_state.initialized = True
    st.session_state.customer_url = ""

# Always sync with URL on reload
url_data = get_storage_data()
if url_data.get('requests'):
    # Merge requests, keeping vendor responses
    for url_req in url_data['requests']:
        found = False
        for idx, session_req in enumerate(st.session_state.requests):
            if session_req['request_id'] == url_req['request_id']:
                # Keep our responses but update request details
                url_req['responses'] = session_req.get('responses', [])
                st.session_state.requests[idx] = url_req
                found = True
                break
        if not found:
            st.session_state.requests.append(url_req)

# Custom CSS with animation
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #A23B72;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .alert-box {
        padding: 2.5rem;
        background: linear-gradient(45deg, #ff6b6b, #ee5a6f, #ff6b6b, #ee5a6f);
        background-size: 400% 400%;
        animation: gradient-shift 3s ease infinite, zigzag 1s ease-in-out infinite;
        border-radius: 20px;
        text-align: center;
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.5);
        transform-origin: center;
        border: 4px solid white;
    }
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        25% { background-position: 100% 50%; }
        50% { background-position: 0% 50%; }
        75% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes zigzag {
        0%, 100% { 
            transform: scale(1) rotate(0deg); 
        }
        25% { 
            transform: scale(1.08) rotate(2deg); 
        }
        50% { 
            transform: scale(1.15) rotate(0deg); 
        }
        75% { 
            transform: scale(1.08) rotate(-2deg); 
        }
    }
    .pending-badge {
        background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
        color: #000;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(255, 193, 7, 0.4);
    }
    .responded-badge {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.4);
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
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏪 MediFind - Vendor Dashboard</h1>', unsafe_allow_html=True)

# URL Import Section
st.sidebar.title("⚙️ Import Customer Requests")
st.sidebar.markdown("### 📥 Paste Customer URL")

customer_url_input = st.sidebar.text_area(
    "Customer URL",
    value=st.session_state.customer_url,
    placeholder="Paste the complete URL from customer app here...",
    height=100,
    help="Customer will share their URL - paste it here"
)

if st.sidebar.button("🔗 Import from URL", type="primary"):
    if customer_url_input.strip():
        try:
            # Extract data parameter from URL
            if '?data=' in customer_url_input:
                data_part = customer_url_input.split('?data=')[1]
                # Decode URL encoded data if needed
                import urllib.parse
                decoded_data = urllib.parse.unquote(data_part)
                imported_data = json.loads(decoded_data)
                
                # Merge with existing requests
                for new_req in imported_data.get('requests', []):
                    exists = False
                    for idx, existing_req in enumerate(st.session_state.requests):
                        if existing_req['request_id'] == new_req['request_id']:
                            # Keep existing responses
                            new_req['responses'] = existing_req.get('responses', [])
                            st.session_state.requests[idx] = new_req
                            exists = True
                            break
                    if not exists:
                        st.session_state.requests.append(new_req)
                
                st.session_state.customer_url = customer_url_input
                save_storage_data({'requests': st.session_state.requests})
                st.sidebar.success("✅ Requests imported!")
                st.rerun()
            else:
                st.sidebar.error("❌ Invalid URL format. URL should contain '?data='")
        except Exception as e:
            st.sidebar.error(f"❌ Error importing: {str(e)}")
    else:
        st.sidebar.warning("⚠️ Please paste customer URL")

st.sidebar.markdown("---")
st.sidebar.info("💡 Customer shares URL → Paste here → Auto-sync!")

# Alert for new requests
pending_count = count_pending_requests(st.session_state.requests)
if pending_count > 0:
    st.markdown(f'''
    <div class="alert-box">
        🚨 {pending_count} NEW REQUEST{"S" if pending_count > 1 else ""}! 🚨
    </div>
    ''', unsafe_allow_html=True)

# Statistics Dashboard
st.markdown("### 📊 Dashboard Statistics")
col1, col2, col3, col4 = st.columns(4)

total_requests = len(st.session_state.requests)
responded = sum(1 for req in st.session_state.requests if req.get('responses'))
pending = pending_count
response_rate = (responded / total_requests * 100) if total_requests > 0 else 0

with col1:
    st.metric("📋 Total Requests", total_requests)
with col2:
    st.metric("✅ Responded", responded)
with col3:
    st.metric("⏳ Pending", pending)
with col4:
    st.metric("📈 Response Rate", f"{response_rate:.0f}%")

# Display requests
st.markdown("---")
st.markdown("### 📋 Medicine Requests")

if not st.session_state.requests:
    st.info("📭 No requests yet. Import customer URL from sidebar to see requests!")
    st.markdown("""
    ### 🔗 How to Get Requests:
    1. Customer submits request in their app
    2. Customer copies their page URL
    3. Customer shares URL with you (WhatsApp, Email, etc.)
    4. You paste URL in sidebar
    5. Click "Import from URL"
    6. Requests appear here instantly!
    """)
else:
    # Sort: pending first
    sorted_requests = sorted(
        st.session_state.requests, 
        key=lambda x: (bool(x.get('responses')), x['timestamp']), 
        reverse=False
    )
    
    for request in sorted_requests:
        request_id = request['request_id']
        has_response = bool(request.get('responses'))
        
        status_badge = '<span class="responded-badge">✅ RESPONDED</span>' if has_response else '<span class="pending-badge">⏳ PENDING</span>'
        
        with st.expander(
            f"{'✅' if has_response else '🔔'} {request['medicine_name']} ({request['medicine_mg']}) - {request_id}",
            expanded=not has_response
        ):
            st.markdown(status_badge, unsafe_allow_html=True)
            st.markdown("")
            
            col_a, col_b = st.columns([2, 1])
            
            with col_a:
                st.markdown(f"**💊 Medicine:** {request['medicine_name']}")
                st.markdown(f"**⚖️ Dosage:** {request['medicine_mg']}")
                st.markdown(f"**🕒 Request Time:** {request['timestamp']}")
                st.markdown(f"**📝 Request ID:** {request_id}")
                
                # Display image
                if request.get('image_data'):
                    try:
                        img_bytes = base64.b64decode(request['image_data'])
                        img = Image.open(BytesIO(img_bytes))
                        st.image(img, caption="Medicine Image", width=300)
                    except:
                        st.warning("Unable to display image")
            
            with col_b:
                if has_response:
                    st.success("✅ Responded")
                else:
                    st.warning("⏰ Needs response")
            
            # Show existing responses
            if has_response:
                st.markdown("---")
                st.markdown("**📤 Your Response:**")
                for resp in request['responses']:
                    st.info(f"""
                    🏪 **Shop:** {resp['shop_name']}  
                    📍 **Address:** {resp['address']}  
                    📞 **Contact:** {resp.get('contact', 'N/A')}  
                    🗺️ **Location:** {resp['location']}
                    """)
            
            # Response form
            if not has_response:
                st.markdown("---")
                st.markdown("### 📤 Send Response to Customer")
                
                with st.form(key=f"form_{request_id}"):
                    shop_name = st.text_input("🏪 Shop Name *", key=f"shop_{request_id}")
                    address = st.text_area("📍 Shop Address *", key=f"addr_{request_id}", height=100)
                    contact = st.text_input("📞 Contact Number", key=f"contact_{request_id}")
                    location = st.text_input("🗺️ GPS Location *", key=f"loc_{request_id}",
                                            placeholder="Google Maps link or coordinates")
                    
                    st.caption("💡 Get location: Google Maps → Long press → Copy coordinates")
                    
                    submit = st.form_submit_button("✉️ Send Response", type="primary", use_container_width=True)
                    
                    if submit:
                        if not shop_name or not address or not location:
                            st.error("⚠️ Fill all required fields!")
                        else:
                            response_data = {
                                'shop_name': shop_name,
                                'address': address,
                                'contact': contact or 'Not provided',
                                'location': location,
                                'response_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            
                            # Update request
                            for req in st.session_state.requests:
                                if req['request_id'] == request_id:
                                    req['responses'].append(response_data)
                                    req['status'] = 'responded'
                                    break
                            
                            # Save to URL
                            save_storage_data({'requests': st.session_state.requests})
                            
                            st.success("✅ Response sent! Customer will see it automatically.")
                            st.balloons()
                            import time
                            time.sleep(1)
                            st.rerun()

# Share response URL
if st.session_state.requests and any(req.get('responses') for req in st.session_state.requests):
    st.markdown("---")
    st.markdown("### 🔗 Share Response with Customer")
    st.info("📋 Copy the current page URL from browser and send to customer. They'll paste it in their app to see your response!")
    st.code("Current URL is in your browser address bar - copy and share with customer")

# Clear option
st.markdown("---")
if st.button("🗑️ Clear All Requests"):
    st.session_state.requests = []
    st.query_params.clear()
    st.success("✅ Cleared!")
    st.rerun()

# Auto-refresh
st.markdown("---")
col_r1, col_r2 = st.columns([3, 1])
with col_r1:
    st.info("💡 Auto-refresh ON - New requests appear automatically!")
with col_r2:
    if st.button("🔄 Refresh"):
        st.rerun()

auto_refresh = st.checkbox("🔄 Auto-refresh (every 5 seconds)", value=True)
if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()

st.markdown("---")
st.caption("🏪 MediFind Vendor | Real-time via URL Sync")

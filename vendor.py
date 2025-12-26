import streamlit as st
import json
import os
from datetime import datetime, timedelta
from PIL import Image
import base64
from io import BytesIO

# Set page config
st.set_page_config(page_title="MediFind - Vendor", page_icon="🏪", layout="wide")

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

def count_pending_requests(data):
    """Count requests without responses"""
    return sum(1 for req in data['requests'] if not req.get('responses'))

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
    .request-card {
        padding: 1.5rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 3px solid #dee2e6;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏪 MediFind - Vendor Dashboard</h1>', unsafe_allow_html=True)

# Load and cleanup data
data = load_data()
data = cleanup_old_requests(data)
save_data(data)

# Show debug info
with st.expander("🔧 Debug Info (Click to expand)"):
    st.code(f"Data file location: {DATA_FILE}")
    st.code(f"File exists: {os.path.exists(DATA_FILE)}")
    st.code(f"File writable: {os.access(DATA_FILE, os.W_OK) if os.path.exists(DATA_FILE) else 'N/A'}")
    st.code(f"Total requests in file: {len(data['requests'])}")
    st.code(f"Pending requests: {pending_count}")
    if st.button("View Raw Data"):
        st.json(data)
    if st.button("Force Refresh Data"):
        st.rerun()

# Alert for new requests
pending_count = count_pending_requests(data)
if pending_count > 0:
    st.markdown(f'''
    <div class="alert-box">
        🚨 {pending_count} NEW REQUEST{"S" if pending_count > 1 else ""}! 🚨
    </div>
    ''', unsafe_allow_html=True)

# Statistics Dashboard
st.markdown("### 📊 Dashboard Statistics")
col1, col2, col3, col4 = st.columns(4)

total_requests = len(data['requests'])
responded = sum(1 for req in data['requests'] if req.get('responses'))
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

if not data['requests']:
    st.info("📭 No requests yet. Waiting for customers to submit requests...")
    st.markdown("💡 **Tip:** Keep auto-refresh enabled to catch new requests instantly!")
else:
    # Sort: pending first, then by timestamp
    sorted_requests = sorted(
        data['requests'], 
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
            # Status badge
            st.markdown(status_badge, unsafe_allow_html=True)
            st.markdown("")
            
            col_a, col_b = st.columns([2, 1])
            
            with col_a:
                st.markdown(f"**💊 Medicine:** {request['medicine_name']}")
                st.markdown(f"**⚖️ Dosage:** {request['medicine_mg']}")
                st.markdown(f"**🕒 Request Time:** {request['timestamp']}")
                st.markdown(f"**📝 Status:** {'Responded' if has_response else 'Awaiting Response'}")
                
                # Display image if available
                if request.get('image_data'):
                    try:
                        img_bytes = base64.b64decode(request['image_data'])
                        img = Image.open(BytesIO(img_bytes))
                        st.image(img, caption="Medicine Image", width=300)
                    except:
                        st.warning("Unable to display image")
            
            with col_b:
                if has_response:
                    st.success(f"✅ Response sent at\n{request['responses'][0]['response_time']}")
                else:
                    st.warning("⏰ No response yet\n\nPlease respond below")
            
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
            
            # Response form (only for pending requests)
            if not has_response:
                st.markdown("---")
                st.markdown("### 📤 Send Your Response")
                
                with st.form(key=f"form_{request_id}"):
                    shop_name = st.text_input("🏪 Shop Name *", key=f"shop_{request_id}", 
                                             placeholder="Enter your shop name")
                    
                    address = st.text_area("📍 Shop Address *", key=f"addr_{request_id}",
                                          placeholder="Enter complete address with landmarks",
                                          height=100)
                    
                    contact = st.text_input("📞 Contact Number", key=f"contact_{request_id}",
                                           placeholder="Your phone number (optional)")
                    
                    st.markdown("**🗺️ GPS Location** (Required)")
                    location = st.text_input("Location Link or Coordinates *", 
                                            key=f"loc_{request_id}",
                                            placeholder="Paste Google Maps link or coordinates")
                    
                    st.caption("💡 **How to get GPS location:**")
                    st.caption("• Open Google Maps → Long press your shop → Click coordinates to copy")
                    st.caption("• Or use 'Share' button → Copy link")
                    
                    col_submit, col_cancel = st.columns([1, 1])
                    
                    with col_submit:
                        submit = st.form_submit_button("✉️ Send Response", 
                                                      type="primary", 
                                                      use_container_width=True)
                    
                    if submit:
                        if not shop_name or not address or not location:
                            st.error("⚠️ Please fill all required fields marked with *")
                        else:
                            # Create response
                            response_data = {
                                'shop_name': shop_name,
                                'address': address,
                                'contact': contact if contact else 'Not provided',
                                'location': location,
                                'response_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            
                            # Find and update request
                            for req in data['requests']:
                                if req['request_id'] == request_id:
                                    req['responses'].append(response_data)
                                    req['status'] = 'responded'
                                    break
                            
                            save_data(data)
                            
                            st.success("✅ Response sent successfully!")
                            st.balloons()
                            import time
                            time.sleep(1)
                            st.rerun()

# Auto-refresh controls
st.markdown("---")
col_refresh1, col_refresh2 = st.columns([3, 1])

with col_refresh1:
    st.info("💡 **Keep this page open!** Enable auto-refresh to catch new requests instantly.")

with col_refresh2:
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()

auto_refresh = st.checkbox("🔄 Auto-refresh every 5 seconds", value=True)

if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()

# Footer
st.markdown("---")
st.caption("🏪 MediFind Vendor Dashboard | Requests auto-delete after 24 hours | Stay online to serve customers!")

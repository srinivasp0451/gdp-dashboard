import streamlit as st
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO
import json

# Set page config
st.set_page_config(page_title="MediFind - Vendor", page_icon="🏪", layout="wide")

# Initialize session state for in-memory storage
if 'requests' not in st.session_state:
    st.session_state.requests = []

def count_pending_requests():
    """Count requests without responses"""
    return sum(1 for req in st.session_state.requests if not req.get('responses'))

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
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏪 MediFind - Vendor Dashboard</h1>', unsafe_allow_html=True)

# Customer API URL input
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("### Customer App URL")
customer_url = st.sidebar.text_input(
    "Enter Customer App URL",
    placeholder="https://your-customer-app.streamlit.app",
    help="The URL where your customer app is hosted"
)
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Get the customer URL from your hosting platform after deploying customer_app.py")

# Alert for new requests
pending_count = count_pending_requests()
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

# Import Request Section
st.markdown("---")
st.markdown("### 📥 Import Customer Request")

col_import1, col_import2 = st.columns([2, 1])

with col_import1:
    st.markdown("#### Paste Request JSON from Customer")
    request_json_input = st.text_area(
        "Request Data",
        height=200,
        placeholder='Paste the complete JSON request from customer here...\n{\n  "request_id": "REQ_...",\n  "medicine_name": "...",\n  ...\n}'
    )
    
    if st.button("➕ Import Request", type="primary", use_container_width=True):
        if request_json_input.strip():
            try:
                new_request = json.loads(request_json_input)
                
                # Check if request already exists
                existing_ids = [req['request_id'] for req in st.session_state.requests]
                if new_request['request_id'] in existing_ids:
                    st.warning("⚠️ This request already exists!")
                else:
                    st.session_state.requests.append(new_request)
                    st.success(f"✅ Request imported: {new_request['medicine_name']}")
                    st.balloons()
                    st.rerun()
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON format. Please check the request data.")
        else:
            st.error("⚠️ Please paste request data")

with col_import2:
    st.info("""
    **📋 How to Import:**
    
    1. Customer submits request
    2. Customer copies JSON data
    3. Paste JSON in text area
    4. Click "Import Request"
    5. Respond to the request
    """)

# Upload JSON file option
uploaded_request = st.file_uploader("Or upload request JSON file", type=['json'], key="upload_request")
if uploaded_request:
    try:
        uploaded_data = json.load(uploaded_request)
        if st.button("Import from File"):
            # Check if it's a single request or multiple
            if isinstance(uploaded_data, dict) and 'request_id' in uploaded_data:
                # Single request
                existing_ids = [req['request_id'] for req in st.session_state.requests]
                if uploaded_data['request_id'] not in existing_ids:
                    st.session_state.requests.append(uploaded_data)
                    st.success("✅ Request imported from file!")
                    st.rerun()
            elif isinstance(uploaded_data, dict) and 'requests' in uploaded_data:
                # Multiple requests
                for req in uploaded_data['requests']:
                    existing_ids = [r['request_id'] for r in st.session_state.requests]
                    if req['request_id'] not in existing_ids:
                        st.session_state.requests.append(req)
                st.success(f"✅ Imported {len(uploaded_data['requests'])} requests!")
                st.rerun()
    except:
        st.error("❌ Invalid JSON file")

# Display requests
st.markdown("---")
st.markdown("### 📋 Medicine Requests")

if not st.session_state.requests:
    st.info("📭 No requests yet. Import customer requests above to get started!")
    st.markdown("💡 **Tip:** Ask customers to share their request JSON with you")
else:
    # Sort: pending first, then by timestamp
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
            # Status badge
            st.markdown(status_badge, unsafe_allow_html=True)
            st.markdown("")
            
            col_a, col_b = st.columns([2, 1])
            
            with col_a:
                st.markdown(f"**💊 Medicine:** {request['medicine_name']}")
                st.markdown(f"**⚖️ Dosage:** {request['medicine_mg']}")
                st.markdown(f"**🕒 Request Time:** {request['timestamp']}")
                st.markdown(f"**📝 Request ID:** {request_id}")
                
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
                    st.success(f"✅ Response sent")
                else:
                    st.warning("⏰ Awaiting response")
            
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
                    ⏰ **Time:** {resp['response_time']}
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
                    
                    submit = st.form_submit_button("✉️ Generate Response", 
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
                            
                            # Update request
                            for req in st.session_state.requests:
                                if req['request_id'] == request_id:
                                    req['responses'].append(response_data)
                                    req['status'] = 'responded'
                                    break
                            
                            st.success("✅ Response created!")
                            st.markdown("### 📋 Copy this JSON and send to customer:")
                            st.code(json.dumps(response_data, indent=2), language='json')
                            
                            # Download button
                            st.download_button(
                                label="⬇️ Download Response JSON",
                                data=json.dumps(response_data, indent=2),
                                file_name=f"response_{request_id}.json",
                                mime="application/json"
                            )
                            
                            st.balloons()
                            import time
                            time.sleep(2)
                            st.rerun()

# Export all responses
st.markdown("---")
st.markdown("### 📤 Export All Responses")

if st.button("Export All Responses as JSON"):
    all_responses = []
    for req in st.session_state.requests:
        if req.get('responses'):
            all_responses.append({
                'request_id': req['request_id'],
                'medicine_name': req['medicine_name'],
                'responses': req['responses']
            })
    
    export_data = {
        'responses': all_responses,
        'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_responses': len(all_responses)
    }
    
    st.download_button(
        label="⬇️ Download All Responses",
        data=json.dumps(export_data, indent=2),
        file_name=f"all_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# Clear data option
st.markdown("---")
if st.button("🗑️ Clear All Requests", type="secondary"):
    st.session_state.requests = []
    st.success("✅ All requests cleared!")
    st.rerun()

# Auto-refresh controls
st.markdown("---")
col_refresh1, col_refresh2 = st.columns([3, 1])

with col_refresh1:
    st.info("💡 **Keep this page open!** Enable auto-refresh to catch new requests.")

with col_refresh2:
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()

auto_refresh = st.checkbox("🔄 Auto-refresh every 10 seconds", value=False)

if auto_refresh:
    import time
    time.sleep(10)
    st.rerun()

# Footer
st.markdown("---")
st.caption("🏪 MediFind Vendor Dashboard | API-Based Communication | Session Storage")

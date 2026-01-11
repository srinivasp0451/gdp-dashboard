import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="AI Interview Assistant",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: 600;
        font-size: 18px;
    }
    .listening {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
        animation: pulse 2s infinite;
    }
    .stopped {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .question-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    .answer-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    #transcriptBox {
        background-color: #fff;
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        min-height: 100px;
        font-size: 16px;
        line-height: 1.6;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False
if 'last_processed' not in st.session_state:
    st.session_state.last_processed = ""

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.markdown("### Speech Recognition Settings")
    trigger_keyword = st.text_input(
        "Trigger Keyword",
        value="I understood",
        help="Say this keyword to process your answer"
    ).lower().strip()
    
    language = st.selectbox(
        "Language",
        options=[
            ("English (US)", "en-US"),
            ("English (UK)", "en-GB"),
            ("Spanish", "es-ES"),
            ("French", "fr-FR"),
            ("German", "de-DE"),
            ("Italian", "it-IT"),
            ("Portuguese", "pt-BR"),
            ("Hindi", "hi-IN"),
            ("Chinese", "zh-CN"),
            ("Japanese", "ja-JP")
        ],
        format_func=lambda x: x[0],
        index=0
    )
    
    continuous_mode = st.checkbox(
        "Continuous Listening",
        value=True,
        help="Keep listening after processing each answer"
    )
    
    st.markdown("### Display Settings")
    show_timestamps = st.checkbox("Show Timestamps", value=True)
    show_raw_transcript = st.checkbox("Show Raw Transcript", value=True)
    
    st.markdown("---")
    st.markdown("### Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions", len(st.session_state.qa_history))
    with col2:
        status = "🟢 Active" if st.session_state.is_listening else "🔴 Stopped"
        st.metric("Status", status)
    
    if st.button("🗑️ Clear All History"):
        st.session_state.qa_history = []
        st.session_state.last_processed = ""
        st.rerun()

# Main content
st.title("🎤 AI Interview Assistant")
st.markdown("### Real-time Speech-to-Text Interview Helper")
st.markdown("*Uses browser's built-in speech recognition - no additional installation required!*")

# Control buttons
col1, col2 = st.columns(2)

with col1:
    start_clicked = st.button("▶️ Start Interview", type="primary", use_container_width=True)
    if start_clicked:
        st.session_state.is_listening = True

with col2:
    stop_clicked = st.button("⏹️ Stop Interview", use_container_width=True)
    if stop_clicked:
        st.session_state.is_listening = False

# Web Speech API Component
speech_component = f"""
<div>
    <div id="statusIndicator" class="status-box stopped">⏸️ Click 'Start Interview' to begin</div>
    <div id="transcriptBox" style="display: {'block' if show_raw_transcript else 'none'};">
        <strong>Live Transcript:</strong><br>
        <span id="transcript">Waiting to start...</span>
    </div>
</div>

<script>
let recognition = null;
let isListening = {json.dumps(st.session_state.is_listening)};
let triggerKeyword = "{trigger_keyword}";
let currentTranscript = "";
let finalTranscript = "";
let language = "{language[1]}";
let continuousMode = {json.dumps(continuous_mode)};

function initializeSpeechRecognition() {{
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {{
        document.getElementById('statusIndicator').innerHTML = '❌ Speech Recognition not supported in this browser. Please use Chrome, Edge, or Safari.';
        document.getElementById('statusIndicator').className = 'status-box stopped';
        return;
    }}
    
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = language;
    
    recognition.onstart = function() {{
        document.getElementById('statusIndicator').innerHTML = '🎙️ LISTENING - Say your trigger keyword after answering';
        document.getElementById('statusIndicator').className = 'status-box listening';
        document.getElementById('transcript').innerHTML = 'Listening...';
    }};
    
    recognition.onresult = function(event) {{
        currentTranscript = "";
        let interimTranscript = "";
        
        for (let i = event.resultIndex; i < event.results.length; i++) {{
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {{
                finalTranscript += transcript + " ";
                currentTranscript = finalTranscript;
            }} else {{
                interimTranscript += transcript;
            }}
        }}
        
        const displayText = currentTranscript + "<i style='color: #666;'>" + interimTranscript + "</i>";
        document.getElementById('transcript').innerHTML = displayText || 'Listening...';
        
        // Check for trigger keyword
        const fullText = (currentTranscript + interimTranscript).toLowerCase();
        if (fullText.includes(triggerKeyword)) {{
            processAnswer(currentTranscript);
        }}
    }};
    
    recognition.onerror = function(event) {{
        console.error('Speech recognition error:', event.error);
        if (event.error === 'no-speech') {{
            document.getElementById('transcript').innerHTML = '⚠️ No speech detected. Please try again.';
        }} else if (event.error === 'not-allowed') {{
            document.getElementById('statusIndicator').innerHTML = '❌ Microphone access denied. Please allow microphone access.';
            document.getElementById('statusIndicator').className = 'status-box stopped';
        }}
    }};
    
    recognition.onend = function() {{
        if (isListening && continuousMode) {{
            // Restart recognition if still in listening mode
            setTimeout(() => {{
                if (isListening) {{
                    recognition.start();
                }}
            }}, 100);
        }} else {{
            document.getElementById('statusIndicator').innerHTML = '⏸️ INTERVIEW STOPPED';
            document.getElementById('statusIndicator').className = 'status-box stopped';
        }}
    }};
}}

function processAnswer(transcript) {{
    // Extract answer before trigger keyword
    const lowerTranscript = transcript.toLowerCase();
    const keywordIndex = lowerTranscript.indexOf(triggerKeyword);
    
    if (keywordIndex > 0) {{
        const answer = transcript.substring(0, keywordIndex).trim();
        if (answer.length > 0) {{
            // Send data to Streamlit
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue',
                value: {{
                    answer: answer,
                    timestamp: new Date().toISOString()
                }}
            }}, '*');
            
            // Visual feedback
            document.getElementById('transcript').innerHTML = 
                '✅ <strong>Answer captured!</strong><br>Ready for next question...';
            
            // Reset for next question
            finalTranscript = "";
            currentTranscript = "";
            
            // Brief pause before continuing
            if (continuousMode) {{
                setTimeout(() => {{
                    document.getElementById('transcript').innerHTML = 'Listening...';
                }}, 2000);
            }}
        }}
    }}
}}

function startListening() {{
    isListening = true;
    if (recognition) {{
        finalTranscript = "";
        currentTranscript = "";
        recognition.start();
    }}
}}

function stopListening() {{
    isListening = false;
    if (recognition) {{
        recognition.stop();
    }}
    document.getElementById('statusIndicator').innerHTML = '⏸️ INTERVIEW STOPPED';
    document.getElementById('statusIndicator').className = 'status-box stopped';
    document.getElementById('transcript').innerHTML = 'Stopped';
}}

// Initialize on load
initializeSpeechRecognition();

// Start/stop based on initial state
if (isListening) {{
    startListening();
}}

// Listen for updates from Streamlit
window.addEventListener('message', function(event) {{
    if (event.data.type === 'streamlit:render') {{
        const newListening = event.data.args.is_listening;
        if (newListening && !isListening) {{
            startListening();
        }} else if (!newListening && isListening) {{
            stopListening();
        }}
    }}
}});
</script>
"""

# Render the speech component
result = components.html(
    speech_component,
    height=200 if show_raw_transcript else 100,
)

# Process received data
if result is not None:
    try:
        if isinstance(result, dict):
            answer = result.get('answer', '')
            timestamp = result.get('timestamp', '')
            
            # Avoid duplicate processing
            if answer and answer != st.session_state.last_processed:
                st.session_state.last_processed = answer
                
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_time = dt.strftime("%H:%M:%S")
                except:
                    formatted_time = datetime.now().strftime("%H:%M:%S")
                
                st.session_state.qa_history.append({
                    'question': f"Question {len(st.session_state.qa_history) + 1}",
                    'answer': answer,
                    'timestamp': formatted_time
                })
                st.rerun()
    except Exception as e:
        pass  # Silently ignore processing errors

# Q&A History
if st.session_state.qa_history:
    st.markdown("---")
    st.markdown("### 📚 Interview Q&A History")
    
    for idx, qa in enumerate(reversed(st.session_state.qa_history), 1):
        with st.expander(
            f"Question {len(st.session_state.qa_history) - idx + 1}" + 
            (f" - {qa['timestamp']}" if show_timestamps else ""), 
            expanded=(idx==1)
        ):
            st.markdown(
                f'<div class="question-box"><strong>❓ Question:</strong> {qa["question"]}</div>', 
                unsafe_allow_html=True
            )
            
            st.markdown(
                f'<div class="answer-box"><strong>✅ Your Answer:</strong><br>{qa["answer"]}</div>', 
                unsafe_allow_html=True
            )
            
            # Editable answer text area
            edited_answer = st.text_area(
                "Formatted Answer (Edit/Copy)",
                value=qa['answer'],
                height=120,
                key=f"answer_{len(st.session_state.qa_history) - idx + 1}"
            )
            
            col_a, col_b = st.columns([3, 1])
            with col_b:
                if st.button(f"🗑️ Delete", key=f"del_{len(st.session_state.qa_history) - idx + 1}"):
                    st.session_state.qa_history.pop(len(st.session_state.qa_history) - idx)
                    st.rerun()
else:
    st.info("👆 Start the interview and begin answering questions. Your responses will appear here!")

# Instructions
with st.expander("ℹ️ How to Use", expanded=False):
    st.markdown("""
    ### Instructions:
    
    1. **Configure Settings** in the sidebar:
       - Set your trigger keyword (default: "I understood")
       - Choose your language
       - Toggle continuous listening mode
       - Customize display preferences
    
    2. **Start Interview**:
       - Click the "▶️ Start Interview" button
       - **Allow microphone access** when your browser prompts you
       - Wait for "LISTENING" status
    
    3. **Answer Questions**:
       - Listen to the interviewer's question
       - Speak your answer clearly into your microphone
       - Say the trigger keyword when done (e.g., "I understood")
       - Your answer will be automatically captured and formatted
       - The system continues listening for the next question
    
    4. **Review & Edit**:
       - Expand any Q&A to view full details
       - Edit answers directly in the text area
       - Copy formatted answers for use elsewhere
       - Delete individual Q&As if needed
    
    5. **Stop Interview**:
       - Click "⏹️ Stop Interview" when done
       - Use "🗑️ Clear All History" to start fresh
    
    ### Tips:
    - **Speak clearly** and at a moderate pace
    - **Minimize background noise** for better recognition
    - Use **Chrome, Edge, or Safari** for best compatibility
    - **Test your microphone** before important interviews
    - The system works **entirely in your browser** - no server processing!
    
    ### Troubleshooting:
    - If speech isn't recognized, check microphone permissions in browser settings
    - Refresh the page if recognition stops working
    - Ensure you're using a supported browser (Chrome recommended)
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "💡 <strong>Browser-based Speech Recognition</strong> - Works without PyAudio! | "
    "Powered by Web Speech API"
    "</div>",
    unsafe_allow_html=True
)

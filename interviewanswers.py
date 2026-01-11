import streamlit as st
import speech_recognition as sr
import threading
import queue
import time
from datetime import datetime

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
    }
    .stopped {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
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
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'current_transcript' not in st.session_state:
    st.session_state.current_transcript = ""
if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if 'stop_flag' not in st.session_state:
    st.session_state.stop_flag = False

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.markdown("### Speech Recognition Settings")
    trigger_keyword = st.text_input(
        "Trigger Keyword",
        value="I understood",
        help="Say this keyword to process your answer"
    ).lower()
    
    energy_threshold = st.slider(
        "Microphone Sensitivity",
        min_value=100,
        max_value=4000,
        value=2000,
        step=100,
        help="Lower values = more sensitive"
    )
    
    pause_threshold = st.slider(
        "Pause Duration (seconds)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Time to wait before processing speech"
    )
    
    st.markdown("### Display Settings")
    show_timestamps = st.checkbox("Show Timestamps", value=True)
    auto_clear = st.checkbox("Auto-clear after processing", value=False)
    
    st.markdown("---")
    st.markdown("### Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions", len(st.session_state.qa_history))
    with col2:
        status = "🟢 Active" if st.session_state.is_listening else "🔴 Stopped"
        st.metric("Status", status)

# Main content
st.title("🎤 AI Interview Assistant")
st.markdown("### Real-time Speech-to-Text Interview Helper")

# Control buttons
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("▶️ Start Interview", type="primary", disabled=st.session_state.is_listening):
        st.session_state.is_listening = True
        st.session_state.stop_flag = False
        st.rerun()

with col2:
    if st.button("⏹️ Stop Interview", disabled=not st.session_state.is_listening):
        st.session_state.is_listening = False
        st.session_state.stop_flag = True
        st.rerun()

with col3:
    if st.button("🗑️ Clear History"):
        st.session_state.qa_history = []
        st.session_state.current_transcript = ""
        st.rerun()

# Status indicator
if st.session_state.is_listening:
    st.markdown('<div class="status-box listening">🎙️ LISTENING - Say your trigger keyword after answering</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-box stopped">⏸️ INTERVIEW STOPPED</div>', unsafe_allow_html=True)

# Speech recognition function
def listen_and_transcribe():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = energy_threshold
    recognizer.pause_threshold = pause_threshold
    
    with sr.Microphone() as source:
        st.session_state.current_transcript = "🎤 Adjusting for ambient noise..."
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        while st.session_state.is_listening and not st.session_state.stop_flag:
            try:
                st.session_state.current_transcript = "🎤 Listening..."
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=30)
                
                try:
                    text = recognizer.recognize_google(audio)
                    st.session_state.current_transcript = f"Heard: {text}"
                    
                    # Check for trigger keyword
                    if trigger_keyword in text.lower():
                        # Extract answer (text before trigger keyword)
                        answer_text = text.lower().split(trigger_keyword)[0].strip()
                        
                        if answer_text:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            st.session_state.qa_history.append({
                                'question': f"Question {len(st.session_state.qa_history) + 1}",
                                'answer': answer_text.capitalize(),
                                'timestamp': timestamp
                            })
                            
                            if auto_clear:
                                st.session_state.current_transcript = ""
                            else:
                                st.session_state.current_transcript = f"✅ Answer processed! Ready for next question."
                            
                            time.sleep(0.5)
                    
                except sr.UnknownValueError:
                    st.session_state.current_transcript = "⚠️ Could not understand audio"
                except sr.RequestError as e:
                    st.session_state.current_transcript = f"❌ Error: {str(e)}"
                    
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                st.session_state.current_transcript = f"❌ Error: {str(e)}"
                break

# Live transcription display
st.markdown("### 📝 Live Transcription")
transcript_placeholder = st.empty()

if st.session_state.is_listening:
    transcript_placeholder.info(st.session_state.current_transcript)
    
    # Start listening in a separate thread
    if 'listener_thread' not in st.session_state or not st.session_state.listener_thread.is_alive():
        st.session_state.listener_thread = threading.Thread(target=listen_and_transcribe, daemon=True)
        st.session_state.listener_thread.start()
    
    # Auto-refresh while listening
    time.sleep(0.5)
    st.rerun()
else:
    transcript_placeholder.warning("Click 'Start Interview' to begin")

# Q&A History
if st.session_state.qa_history:
    st.markdown("---")
    st.markdown("### 📚 Interview Q&A History")
    
    for idx, qa in enumerate(reversed(st.session_state.qa_history), 1):
        with st.expander(f"Question {len(st.session_state.qa_history) - idx + 1}" + 
                        (f" - {qa['timestamp']}" if show_timestamps else ""), expanded=(idx==1)):
            
            st.markdown(f'<div class="question-box"><strong>❓ Question:</strong> {qa["question"]}</div>', 
                       unsafe_allow_html=True)
            
            st.markdown(f'<div class="answer-box"><strong>✅ Your Answer:</strong><br>{qa["answer"]}</div>', 
                       unsafe_allow_html=True)
            
            # Answer text area for editing/copying
            st.text_area(
                "Formatted Answer (Copy/Edit)",
                value=qa['answer'],
                height=100,
                key=f"answer_{len(st.session_state.qa_history) - idx + 1}"
            )

# Instructions
with st.expander("ℹ️ How to Use", expanded=False):
    st.markdown("""
    ### Instructions:
    
    1. **Configure Settings** in the sidebar:
       - Set your trigger keyword (default: "I understood")
       - Adjust microphone sensitivity
       - Customize display preferences
    
    2. **Start Interview**:
       - Click the "▶️ Start Interview" button
       - Allow microphone access when prompted
       - Wait for "Listening..." status
    
    3. **Answer Questions**:
       - Listen to the interviewer's question
       - Speak your answer clearly
       - Say the trigger keyword when done (e.g., "I understood")
       - Your answer will be automatically captured and formatted
    
    4. **Review & Edit**:
       - Expand any Q&A to view full details
       - Copy or edit answers as needed
       - Timestamps show when each answer was recorded
    
    5. **Stop Interview**:
       - Click "⏹️ Stop Interview" when done
       - Use "🗑️ Clear History" to start fresh
    
    ### Tips:
    - Speak clearly and at a moderate pace
    - Ensure minimal background noise
    - Adjust sensitivity if recognition is poor
    - Test your microphone before important interviews
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "💡 Tip: Practice with the assistant to improve your interview responses!"
    "</div>",
    unsafe_allow_html=True
)

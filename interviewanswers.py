import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
import json
import requests
from bs4 import BeautifulSoup
import time

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
    .processing {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeaa7;
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
    .source-box {
        background-color: #fff3e0;
        padding: 10px;
        border-radius: 6px;
        border-left: 3px solid #ff9800;
        margin: 5px 0;
        font-size: 14px;
    }
    .source-link {
        color: #1976D2;
        text-decoration: none;
        font-weight: 500;
    }
    .source-link:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False
if 'last_processed_question' not in st.session_state:
    st.session_state.last_processed_question = ""
if 'processing' not in st.session_state:
    st.session_state.processing = False

def search_web(query):
    """Search DuckDuckGo for answers"""
    try:
        # DuckDuckGo Instant Answer API
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1
        }
        
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        results = []
        
        # Get abstract
        if data.get('Abstract'):
            results.append({
                'answer': data['Abstract'],
                'source': data.get('AbstractURL', 'DuckDuckGo'),
                'title': data.get('Heading', 'Answer')
            })
        
        # Get related topics
        if data.get('RelatedTopics'):
            for topic in data['RelatedTopics'][:3]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'answer': topic['Text'],
                        'source': topic.get('FirstURL', 'DuckDuckGo'),
                        'title': 'Related Information'
                    })
        
        return results
    except Exception as e:
        return [{'answer': f'Error searching: {str(e)}', 'source': 'Error', 'title': 'Error'}]

def search_wikipedia(query):
    """Search Wikipedia for answers"""
    try:
        # Wikipedia API
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'utf8': 1,
            'srlimit': 1
        }
        
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        if data.get('query', {}).get('search'):
            result = data['query']['search'][0]
            title = result['title']
            snippet = BeautifulSoup(result['snippet'], 'html.parser').get_text()
            
            # Get full extract
            extract_params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'exintro': 1,
                'explaintext': 1
            }
            
            extract_response = requests.get(url, params=extract_params, timeout=5)
            extract_data = extract_response.json()
            
            pages = extract_data.get('query', {}).get('pages', {})
            if pages:
                page = list(pages.values())[0]
                full_text = page.get('extract', snippet)
                
                # Limit to first 3 sentences
                sentences = full_text.split('. ')[:3]
                answer_text = '. '.join(sentences) + '.'
                
                return [{
                    'answer': answer_text,
                    'source': f'https://en.wikipedia.org/wiki/{title.replace(" ", "_")}',
                    'title': title
                }]
        
        return []
    except Exception as e:
        return []

def get_answer(question):
    """Get answer from multiple sources"""
    all_results = []
    
    # Search Wikipedia first
    wiki_results = search_wikipedia(question)
    all_results.extend(wiki_results)
    
    # Then DuckDuckGo
    ddg_results = search_web(question)
    all_results.extend(ddg_results)
    
    # If no results, provide a generic message
    if not all_results:
        all_results.append({
            'answer': f'No specific answer found for "{question}". Please try rephrasing or check the sources manually.',
            'source': 'No source available',
            'title': 'No Results'
        })
    
    return all_results

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.markdown("### Speech Recognition Settings")
    trigger_keyword = st.text_input(
        "Trigger Keyword",
        value="I understood",
        help="Say this keyword to search for the answer"
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
    
    st.markdown("### Search Settings")
    max_sources = st.slider(
        "Maximum Sources",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of sources to display"
    )
    
    auto_search = st.checkbox(
        "Auto-search on trigger",
        value=True,
        help="Automatically search when trigger keyword is detected"
    )
    
    st.markdown("### Display Settings")
    show_timestamps = st.checkbox("Show Timestamps", value=True)
    show_raw_transcript = st.checkbox("Show Live Transcript", value=True)
    
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
        st.session_state.last_processed_question = ""
        st.rerun()

# Main content
st.title("🎤 AI Interview Assistant with Web Search")
st.markdown("### Ask questions, get answers from the web automatically!")
st.markdown("*Searches Wikipedia and DuckDuckGo for instant answers*")

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
        st.session_state.processing = False

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
let autoSearch = {json.dumps(auto_search)};
let questionDetected = false;

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
        document.getElementById('statusIndicator').innerHTML = '🎙️ LISTENING - Ask your question, then say "' + triggerKeyword + '"';
        document.getElementById('statusIndicator').className = 'status-box listening';
        document.getElementById('transcript').innerHTML = 'Listening...';
    }};
    
    recognition.onresult = function(event) {{
        let interimTranscript = "";
        finalTranscript = "";
        
        for (let i = 0; i < event.results.length; i++) {{
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {{
                finalTranscript += transcript + " ";
            }} else {{
                interimTranscript += transcript;
            }}
        }}
        
        currentTranscript = finalTranscript.trim();
        const cleanTranscript = cleanRepeatedWords(currentTranscript);
        const cleanInterim = cleanRepeatedWords(interimTranscript);
        
        const displayText = cleanTranscript + " <i style='color: #666;'>" + cleanInterim + "</i>";
        document.getElementById('transcript').innerHTML = displayText || 'Listening...';
        
        const fullText = (cleanTranscript + " " + cleanInterim).toLowerCase();
        if (fullText.includes(triggerKeyword) && !questionDetected) {{
            questionDetected = true;
            processQuestion(cleanTranscript);
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
        if (isListening && !questionDetected) {{
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

function processQuestion(transcript) {{
    const cleanedTranscript = cleanRepeatedWords(transcript);
    const lowerTranscript = cleanedTranscript.toLowerCase();
    const keywordIndex = lowerTranscript.indexOf(triggerKeyword);
    
    if (keywordIndex >= 0) {{
        const question = cleanedTranscript.substring(0, keywordIndex).trim();
        
        if (question.length > 5) {{
            document.getElementById('statusIndicator').innerHTML = '🔍 SEARCHING FOR ANSWER...';
            document.getElementById('statusIndicator').className = 'status-box processing';
            
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue',
                value: {{
                    question: question,
                    timestamp: new Date().toISOString(),
                    action: 'search'
                }}
            }}, '*');
            
            document.getElementById('transcript').innerHTML = 
                '✅ <strong>Question detected:</strong><br>"' + question + '"<br>🔍 Searching web for answer...';
            
            setTimeout(() => {{
                finalTranscript = "";
                currentTranscript = "";
                questionDetected = false;
                
                if (isListening) {{
                    document.getElementById('transcript').innerHTML = 'Ready for next question...';
                    document.getElementById('statusIndicator').innerHTML = '🎙️ LISTENING - Ask your next question';
                    document.getElementById('statusIndicator').className = 'status-box listening';
                }}
            }}, 3000);
        }}
    }}
}}

function cleanRepeatedWords(text) {{
    if (!text) return text;
    const words = text.split(' ');
    const cleanedWords = [];
    let prevWord = '';
    
    for (let word of words) {{
        const cleanWord = word.toLowerCase().trim();
        if (cleanWord !== prevWord.toLowerCase().trim() || cleanWord.length < 3) {{
            cleanedWords.push(word);
        }}
        prevWord = word;
    }}
    
    return cleanedWords.join(' ');
}}

function startListening() {{
    isListening = true;
    questionDetected = false;
    if (recognition) {{
        finalTranscript = "";
        currentTranscript = "";
        recognition.start();
    }}
}}

function stopListening() {{
    isListening = false;
    questionDetected = false;
    if (recognition) {{
        recognition.stop();
    }}
    document.getElementById('statusIndicator').innerHTML = '⏸️ INTERVIEW STOPPED';
    document.getElementById('statusIndicator').className = 'status-box stopped';
    document.getElementById('transcript').innerHTML = 'Stopped';
}}

initializeSpeechRecognition();

if (isListening) {{
    startListening();
}}
</script>
"""

# Render the speech component
result = components.html(speech_component, height=200 if show_raw_transcript else 100)

# Process received data and search
if result is not None:
    try:
        if isinstance(result, dict) and result.get('action') == 'search':
            question = result.get('question', '')
            timestamp = result.get('timestamp', '')
            
            if question and question != st.session_state.last_processed_question:
                st.session_state.last_processed_question = question
                st.session_state.processing = True
                
                # Show processing message
                with st.spinner(f"🔍 Searching for answer to: '{question}'"):
                    # Get answers from web
                    answers = get_answer(question)
                    
                    # Limit to max_sources
                    answers = answers[:max_sources]
                    
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        formatted_time = dt.strftime("%H:%M:%S")
                    except:
                        formatted_time = datetime.now().strftime("%H:%M:%S")
                    
                    st.session_state.qa_history.append({
                        'question': question,
                        'answers': answers,
                        'timestamp': formatted_time
                    })
                    
                    st.session_state.processing = False
                    st.success("✅ Answer found and added to history!")
                    time.sleep(1)
                    st.rerun()
    except Exception as e:
        st.error(f"Error processing: {str(e)}")
        st.session_state.processing = False

# Q&A History
if st.session_state.qa_history:
    st.markdown("---")
    st.markdown("### 📚 Interview Q&A History")
    
    for idx, qa in enumerate(reversed(st.session_state.qa_history), 1):
        with st.expander(
            f"❓ {qa['question'][:80]}..." if len(qa['question']) > 80 else f"❓ {qa['question']}" + 
            (f" - {qa['timestamp']}" if show_timestamps else ""), 
            expanded=(idx==1)
        ):
            st.markdown(
                f'<div class="question-box"><strong>❓ Question:</strong><br>{qa["question"]}</div>', 
                unsafe_allow_html=True
            )
            
            # Display all answers with sources
            for ans_idx, answer in enumerate(qa['answers'], 1):
                st.markdown(
                    f'<div class="answer-box">'
                    f'<strong>✅ Answer {ans_idx}:</strong><br>{answer["answer"]}'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    f'<div class="source-box">'
                    f'📎 <strong>Source:</strong> '
                    f'<a href="{answer["source"]}" target="_blank" class="source-link">{answer["title"]}</a> | '
                    f'<a href="{answer["source"]}" target="_blank" class="source-link">{answer["source"][:60]}...</a>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Combined answer for copying
            combined_answer = "\n\n".join([
                f"Answer {i+1}:\n{ans['answer']}\n\nSource: {ans['title']}\n{ans['source']}"
                for i, ans in enumerate(qa['answers'])
            ])
            
            st.text_area(
                "📋 Combined Answer (Copy/Edit)",
                value=combined_answer,
                height=200,
                key=f"answer_{len(st.session_state.qa_history) - idx + 1}"
            )
            
            col_a, col_b = st.columns([3, 1])
            with col_b:
                if st.button(f"🗑️ Delete", key=f"del_{len(st.session_state.qa_history) - idx + 1}"):
                    st.session_state.qa_history.pop(len(st.session_state.qa_history) - idx)
                    st.rerun()
else:
    st.info("👆 Start the interview and ask questions. Answers will be automatically searched from the web!")

# Instructions
with st.expander("ℹ️ How to Use", expanded=False):
    st.markdown("""
    ### Instructions:
    
    1. **Configure Settings** in the sidebar:
       - Set your trigger keyword (default: "I understood")
       - Choose language and max sources
       - Customize display preferences
    
    2. **Start Interview**:
       - Click "▶️ Start Interview"
       - Allow microphone access
    
    3. **Ask Questions**:
       - Speak your question clearly: *"What is polymorphism in Python?"*
       - Say the trigger keyword: *"I understood"*
       - **The app will automatically search the web** for answers
       - Answers from Wikipedia and DuckDuckGo will be displayed with sources
    
    4. **Review Answers**:
       - Each question shows multiple answers with source links
       - Click source links to read full articles
       - Copy formatted answers for your use
    
    5. **Continue or Stop**:
       - Ask more questions - system keeps listening
       - Click "⏹️ Stop Interview" when done
    
    ### Example Flow:
    1. You say: *"Explain recursion in programming I understood"*
    2. App searches web automatically
    3. Displays answers from Wikipedia, DuckDuckGo with clickable source links
    4. You can copy/edit the answers
    5. Ready for next question!
    
    ### Tips:
    - Speak questions clearly before saying trigger keyword
    - System searches Wikipedia and DuckDuckGo automatically
    - Multiple sources give you comprehensive answers
    - Click source links to verify information
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "💡 <strong>Web-Powered Interview Assistant</strong> - Searches Wikipedia & DuckDuckGo automatically! 🌐"
    "</div>",
    unsafe_allow_html=True
)

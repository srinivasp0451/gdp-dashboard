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
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #2196F3;
        margin: 15px 0;
        font-size: 16px;
    }
    .answer-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #4CAF50;
        margin: 15px 0;
        font-size: 16px;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 12px;
        border-radius: 6px;
        border-left: 3px solid #ff9800;
        margin: 8px 0;
        font-size: 14px;
    }
    .source-link {
        color: #1976D2;
        text-decoration: none;
        font-weight: 600;
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
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'should_search' not in st.session_state:
    st.session_state.should_search = False

def clean_repeated_words(text):
    """Remove consecutive duplicate words"""
    if not text:
        return text
    words = text.split()
    cleaned = []
    prev = ""
    for word in words:
        if word.lower() != prev.lower() or len(word) < 3:
            cleaned.append(word)
        prev = word
    return " ".join(cleaned)

def search_web(query):
    """Search DuckDuckGo for answers"""
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        results = []
        
        # Get abstract
        if data.get('Abstract'):
            results.append({
                'answer': data['Abstract'],
                'source': data.get('AbstractURL', 'DuckDuckGo'),
                'title': data.get('Heading', 'DuckDuckGo Answer')
            })
        
        # Get related topics
        if data.get('RelatedTopics'):
            for topic in data['RelatedTopics'][:2]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'answer': topic['Text'],
                        'source': topic.get('FirstURL', 'DuckDuckGo'),
                        'title': 'Related Information'
                    })
        
        return results
    except Exception as e:
        return []

def search_wikipedia(query):
    """Search Wikipedia for answers"""
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'utf8': 1,
            'srlimit': 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('query', {}).get('search'):
            result = data['query']['search'][0]
            title = result['title']
            
            # Get full extract
            extract_params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'exintro': 1,
                'explaintext': 1
            }
            
            extract_response = requests.get(url, params=extract_params, timeout=10)
            extract_data = extract_response.json()
            
            pages = extract_data.get('query', {}).get('pages', {})
            if pages:
                page = list(pages.values())[0]
                full_text = page.get('extract', '')
                
                # Limit to first 4 sentences
                sentences = full_text.split('. ')[:4]
                answer_text = '. '.join(sentences) + '.' if sentences else full_text
                
                return [{
                    'answer': answer_text,
                    'source': f'https://en.wikipedia.org/wiki/{title.replace(" ", "_")}',
                    'title': f'Wikipedia: {title}'
                }]
        
        return []
    except Exception as e:
        return []

def get_answer(question):
    """Get answer from multiple sources"""
    all_results = []
    
    # Search Wikipedia first
    with st.spinner("🔍 Searching Wikipedia..."):
        wiki_results = search_wikipedia(question)
        all_results.extend(wiki_results)
    
    # Then DuckDuckGo
    with st.spinner("🔍 Searching DuckDuckGo..."):
        ddg_results = search_web(question)
        all_results.extend(ddg_results)
    
    # If no results, provide a message
    if not all_results:
        all_results.append({
            'answer': f'No specific answer found online for "{question}". Try rephrasing your question or search manually.',
            'source': 'https://www.google.com/search?q=' + question.replace(' ', '+'),
            'title': 'No Results - Click to search Google'
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
    
    st.markdown("### Display Settings")
    show_timestamps = st.checkbox("Show Timestamps", value=True)
    show_sources = st.checkbox("Show Source Links", value=True)
    
    st.markdown("---")
    st.markdown("### Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions", len(st.session_state.qa_history))
    with col2:
        status = "🟢 Active" if st.session_state.is_listening else "🔴 Stopped"
        st.metric("Status", status)

# Main content
st.title("🎤 AI Interview Assistant with Web Search")
st.markdown("### Ask questions, get instant answers from Wikipedia & DuckDuckGo!")

# Control buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Start Interview", type="primary", use_container_width=True):
        st.session_state.is_listening = True
        st.rerun()

with col2:
    if st.button("⏹️ Stop Interview", use_container_width=True):
        st.session_state.is_listening = False
        st.rerun()

with col3:
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.qa_history = []
        st.session_state.current_question = ""
        st.rerun()

# Status display
if st.session_state.is_listening:
    st.markdown('<div class="status-box listening">🎙️ LISTENING - Ask your question then say your trigger keyword</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-box stopped">⏸️ INTERVIEW STOPPED - Click Start to begin</div>', unsafe_allow_html=True)

# Speech recognition component
speech_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        #transcript {{ 
            background: #f0f0f0; 
            padding: 15px; 
            border-radius: 8px; 
            min-height: 80px;
            font-size: 16px;
            line-height: 1.6;
        }}
        .interim {{ color: #666; font-style: italic; }}
        .detected {{ color: #4CAF50; font-weight: bold; }}
    </style>
</head>
<body>
    <div id="transcript">Ready to listen...</div>
    <script>
        let recognition;
        let finalTranscript = '';
        let isListening = {json.dumps(st.session_state.is_listening)};
        const triggerKeyword = '{trigger_keyword}';
        
        function cleanRepeatedWords(text) {{
            if (!text) return text;
            const words = text.split(' ');
            const cleaned = [];
            let prev = '';
            for (let word of words) {{
                if (word.toLowerCase() !== prev.toLowerCase() || word.length < 3) {{
                    cleaned.push(word);
                }}
                prev = word;
            }}
            return cleaned.join(' ');
        }}
        
        function initSpeech() {{
            if (!('webkitSpeechRecognition' in window)) {{
                document.getElementById('transcript').innerHTML = 
                    '❌ Speech Recognition not supported. Use Chrome/Edge/Safari.';
                return;
            }}
            
            recognition = new webkitSpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = '{language[1]}';
            
            recognition.onstart = () => {{
                document.getElementById('transcript').innerHTML = 
                    '🎙️ Listening... Ask your question and say "<strong>{trigger_keyword}</strong>"';
            }};
            
            recognition.onresult = (event) => {{
                let interim = '';
                finalTranscript = '';
                
                for (let i = 0; i < event.results.length; i++) {{
                    if (event.results[i].isFinal) {{
                        finalTranscript += event.results[i][0].transcript + ' ';
                    }} else {{
                        interim += event.results[i][0].transcript;
                    }}
                }}
                
                const cleaned = cleanRepeatedWords(finalTranscript);
                const cleanedInterim = cleanRepeatedWords(interim);
                
                document.getElementById('transcript').innerHTML = 
                    cleaned + ' <span class="interim">' + cleanedInterim + '</span>';
                
                // Check for trigger
                const fullText = (cleaned + ' ' + cleanedInterim).toLowerCase();
                if (fullText.includes(triggerKeyword.toLowerCase())) {{
                    const keywordIndex = cleaned.toLowerCase().indexOf(triggerKeyword.toLowerCase());
                    if (keywordIndex > 0) {{
                        const question = cleaned.substring(0, keywordIndex).trim();
                        if (question.length > 3) {{
                            document.getElementById('transcript').innerHTML = 
                                '✅ <span class="detected">Question detected: "' + question + '"</span><br>🔍 Searching for answer...';
                            
                            // Send to Streamlit
                            window.parent.postMessage({{
                                type: 'streamlit:setComponentValue',
                                question: question,
                                timestamp: new Date().toISOString()
                            }}, '*');
                            
                            // Reset
                            finalTranscript = '';
                            setTimeout(() => {{
                                if (isListening) {{
                                    document.getElementById('transcript').innerHTML = '🎙️ Ready for next question...';
                                }}
                            }}, 2000);
                        }}
                    }}
                }}
            }};
            
            recognition.onerror = (event) => {{
                if (event.error === 'not-allowed') {{
                    document.getElementById('transcript').innerHTML = 
                        '❌ Microphone access denied. Please allow microphone access.';
                }}
            }};
            
            recognition.onend = () => {{
                if (isListening) {{
                    setTimeout(() => recognition.start(), 100);
                }}
            }};
            
            if (isListening) {{
                recognition.start();
            }}
        }}
        
        initSpeech();
    </script>
</body>
</html>
"""

# Display speech component
question_data = components.html(speech_html, height=150)

# Process question if received
if question_data and isinstance(question_data, dict):
    question = question_data.get('question', '').strip()
    
    if question and question != st.session_state.current_question:
        st.session_state.current_question = question
        st.session_state.should_search = True

# Perform search if needed
if st.session_state.should_search and st.session_state.current_question:
    question = st.session_state.current_question
    
    st.markdown("---")
    st.markdown("### 🔍 Searching for Answer...")
    
    # Display the question
    st.markdown(
        f'<div class="question-box"><strong>❓ YOUR QUESTION:</strong><br><h3>{question}</h3></div>',
        unsafe_allow_html=True
    )
    
    # Search for answers
    answers = get_answer(question)[:max_sources]
    
    if answers:
        st.markdown("### ✅ ANSWERS FROM WEB:")
        
        for idx, answer in enumerate(answers, 1):
            # Display answer
            st.markdown(
                f'<div class="answer-box">'
                f'<strong>📝 Answer {idx}:</strong><br>'
                f'<p style="font-size: 16px; line-height: 1.8;">{answer["answer"]}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Display source
            if show_sources:
                st.markdown(
                    f'<div class="source-box">'
                    f'🔗 <strong>Source:</strong> '
                    f'<a href="{answer["source"]}" target="_blank" class="source-link">{answer["title"]}</a><br>'
                    f'<small>{answer["source"]}</small>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
        # Save to history
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.qa_history.append({
            'question': question,
            'answers': answers,
            'timestamp': timestamp
        })
        
        # Create combined text for copying
        combined = f"QUESTION: {question}\n\n"
        for idx, ans in enumerate(answers, 1):
            combined += f"ANSWER {idx}:\n{ans['answer']}\n\nSOURCE: {ans['title']}\n{ans['source']}\n\n"
        
        st.text_area("📋 Copy All Content", value=combined, height=200, key="current_copy")
        
        st.success("✅ Answer added to history below!")
    
    st.session_state.should_search = False
    st.session_state.current_question = ""

# Display history
if st.session_state.qa_history:
    st.markdown("---")
    st.markdown("## 📚 Interview History")
    
    for idx, qa in enumerate(reversed(st.session_state.qa_history), 1):
        with st.expander(
            f"Question {len(st.session_state.qa_history) - idx + 1}: {qa['question'][:60]}..." 
            if len(qa['question']) > 60 else f"Question {len(st.session_state.qa_history) - idx + 1}: {qa['question']}"
            + (f" ({qa['timestamp']})" if show_timestamps else ""),
            expanded=False
        ):
            # Question
            st.markdown(
                f'<div class="question-box"><strong>❓ QUESTION:</strong><br>{qa["question"]}</div>',
                unsafe_allow_html=True
            )
            
            # Answers
            for ans_idx, answer in enumerate(qa['answers'], 1):
                st.markdown(
                    f'<div class="answer-box">'
                    f'<strong>📝 Answer {ans_idx}:</strong><br>{answer["answer"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                if show_sources:
                    st.markdown(
                        f'<div class="source-box">'
                        f'🔗 <a href="{answer["source"]}" target="_blank" class="source-link">{answer["title"]}</a>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            
            # Combined text
            combined = f"QUESTION: {qa['question']}\n\n"
            for ans_idx, ans in enumerate(qa['answers'], 1):
                combined += f"ANSWER {ans_idx}:\n{ans['answer']}\n\nSOURCE: {ans['source']}\n\n"
            
            st.text_area("📋 Copy", value=combined, height=150, key=f"copy_{len(st.session_state.qa_history) - idx + 1}")

# Instructions
with st.expander("ℹ️ How to Use This App"):
    st.markdown("""
    ### Quick Start:
    
    1. **Click "▶️ Start Interview"** 
    2. **Allow microphone access** when prompted
    3. **Ask your question**: Speak clearly - "What is polymorphism in Python?"
    4. **Say trigger keyword**: "I understood"
    5. **See results**: Question + Answers with sources appear immediately!
    
    ### Example:
    - **You say**: "Explain machine learning I understood"
    - **App shows**: 
        - ❓ YOUR QUESTION: "Explain machine learning"
        - ✅ ANSWERS from Wikipedia & DuckDuckGo with clickable sources
    
    ### Tips:
    - Speak clearly and at moderate pace
    - Wait for "Listening..." status
    - Say full question before trigger keyword
    - Each Q&A is saved in history below
    - Click source links to read full articles
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "💡 Powered by Wikipedia & DuckDuckGo | Browser-based Speech Recognition"
    "</div>",
    unsafe_allow_html=True
)

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
import requests
import re
import json
import time

st.set_page_config(
    page_title="Interview Assistant",
    page_icon="🎤",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .big-font { font-size: 24px !important; font-weight: bold; }
    .stButton>button {
        width: 100%;
        height: 80px;
        font-size: 22px;
        font-weight: bold;
        border-radius: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Session state
if 'qa_list' not in st.session_state:
    st.session_state.qa_list = []
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False
if 'current_transcript' not in st.session_state:
    st.session_state.current_transcript = ""
if 'processing' not in st.session_state:
    st.session_state.processing = False

def clean_repeated_words(text):
    words = text.split()
    result = []
    prev = ""
    for word in words:
        if word.lower() != prev.lower() or len(word) < 3:
            result.append(word)
        prev = word
    return " ".join(result)

def search_wikipedia(query):
    try:
        url = "https://en.wikipedia.org/w/api.php"
        
        search_params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'utf8': 1,
            'srlimit': 2
        }
        
        search_resp = requests.get(url, params=search_params, timeout=10)
        search_data = search_resp.json()
        
        results = []
        for item in search_data.get('query', {}).get('search', [])[:1]:
            title = item['title']
            
            extract_params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'exintro': 1,
                'explaintext': 1
            }
            
            extract_resp = requests.get(url, params=extract_params, timeout=10)
            extract_data = extract_resp.json()
            
            pages = extract_data.get('query', {}).get('pages', {})
            if pages:
                page = list(pages.values())[0]
                extract = page.get('extract', '')
                
                if extract:
                    sentences = re.split(r'(?<=[.!?])\s+', extract)[:5]
                    answer_text = ' '.join(sentences)
                    
                    results.append({
                        'answer': answer_text,
                        'source': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        'title': f'Wikipedia: {title}'
                    })
        
        return results
    except:
        return []

def search_duckduckgo(query):
    try:
        url = "https://api.duckduckgo.com/"
        params = {'q': query, 'format': 'json', 'no_html': 1, 'skip_disambig': 1}
        
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        results = []
        
        if data.get('Abstract'):
            results.append({
                'answer': data['Abstract'],
                'source': data.get('AbstractURL', 'https://duckduckgo.com'),
                'title': data.get('Heading', 'DuckDuckGo')
            })
        
        for topic in data.get('RelatedTopics', [])[:1]:
            if isinstance(topic, dict) and topic.get('Text'):
                results.append({
                    'answer': topic['Text'],
                    'source': topic.get('FirstURL', 'https://duckduckgo.com'),
                    'title': 'DuckDuckGo Related'
                })
        
        return results
    except:
        return []

def get_answers(question):
    all_answers = []
    
    wiki_results = search_wikipedia(question)
    all_answers.extend(wiki_results)
    
    ddg_results = search_duckduckgo(question)
    all_answers.extend(ddg_results)
    
    if not all_answers:
        all_answers.append({
            'answer': f'No results found for "{question}". Try rephrasing your question.',
            'source': f'https://www.google.com/search?q={question.replace(" ", "+")}',
            'title': 'Search Google'
        })
    
    return all_answers

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    trigger_keyword = st.text_input(
        "Trigger Keyword",
        value="I understood",
        help="Say this keyword ANYTIME after your question to trigger search"
    ).lower().strip()
    
    st.info(f"💡 You can say '{trigger_keyword}' anytime - no rush!")
    
    st.metric("Questions Asked", len(st.session_state.qa_list))
    
    if st.button("🗑️ Clear All History"):
        st.session_state.qa_list = []
        st.session_state.current_transcript = ""
        st.rerun()

# Title
st.title("🎤 Interview Assistant")

# Control buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ START LISTENING", type="primary", disabled=st.session_state.is_listening):
        st.session_state.is_listening = True
        st.rerun()

with col2:
    if st.button("⏹️ STOP LISTENING", disabled=not st.session_state.is_listening):
        st.session_state.is_listening = False
        st.rerun()

st.markdown("---")

# Main layout - 2 columns
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("🎙️ LIVE SPEECH CAPTURE")
    
    # Speech recognition component
    speech_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 10px;
                background: #f0f2f6;
                margin: 0;
            }}
            #status {{
                font-size: 20px;
                font-weight: bold;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 15px;
            }}
            .listening {{
                background: #4CAF50;
                color: white;
                animation: pulse 2s infinite;
            }}
            .stopped {{
                background: #f44336;
                color: white;
            }}
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.7; }}
            }}
            #transcript {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                min-height: 200px;
                max-height: 400px;
                overflow-y: auto;
                font-size: 18px;
                line-height: 1.8;
                border: 3px solid #ddd;
            }}
            .detected {{
                background: #e8f5e9;
                padding: 10px;
                border-radius: 5px;
                color: #2e7d32;
                font-weight: bold;
                margin: 10px 0;
            }}
            .interim {{ color: #999; font-style: italic; }}
        </style>
    </head>
    <body>
        <div id="status">Click START LISTENING</div>
        <div id="transcript">Waiting...</div>
        
        <script>
            let recognition;
            let isListening = {json.dumps(st.session_state.is_listening)};
            let fullTranscript = '';
            const triggerWord = '{trigger_keyword}';
            let questionSent = false;
            
            function cleanWords(text) {{
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
            
            function init() {{
                if (!('webkitSpeechRecognition' in window)) {{
                    document.getElementById('status').innerHTML = '❌ Not Supported - Use Chrome/Edge';
                    document.getElementById('status').className = 'stopped';
                    document.getElementById('transcript').innerHTML = 'Speech recognition not supported in this browser. Please use Chrome, Edge, or Safari.';
                    return;
                }}
                
                recognition = new webkitSpeechRecognition();
                recognition.continuous = true;
                recognition.interimResults = true;
                recognition.lang = 'en-US';
                
                recognition.onstart = () => {{
                    document.getElementById('status').innerHTML = '🎙️ LISTENING - Speak freely, say "' + triggerWord + '" when ready';
                    document.getElementById('status').className = 'listening';
                    document.getElementById('transcript').innerHTML = 'Speak now...';
                    questionSent = false;
                    fullTranscript = '';
                }};
                
                recognition.onresult = (event) => {{
                    let interim = '';
                    fullTranscript = '';
                    
                    for (let i = 0; i < event.results.length; i++) {{
                        if (event.results[i].isFinal) {{
                            fullTranscript += event.results[i][0].transcript + ' ';
                        }} else {{
                            interim += event.results[i][0].transcript;
                        }}
                    }}
                    
                    const cleanFinal = cleanWords(fullTranscript);
                    const cleanInterim = cleanWords(interim);
                    
                    // Display everything clearly
                    document.getElementById('transcript').innerHTML = 
                        '<strong>Your Speech:</strong><br><br>' +
                        cleanFinal + 
                        '<span class="interim">' + cleanInterim + '</span>';
                    
                    // Check for trigger ONLY in final transcript
                    if (cleanFinal.toLowerCase().includes(triggerWord) && !questionSent) {{
                        const parts = cleanFinal.toLowerCase().split(triggerWord);
                        const question = parts[0].trim();
                        
                        if (question.length > 5) {{
                            questionSent = true;
                            
                            document.getElementById('transcript').innerHTML = 
                                '<div class="detected">✅ QUESTION CAPTURED!</div>' +
                                '<strong>Question:</strong> ' + question + 
                                '<br><br>🔍 Searching for answer...';
                            
                            // Send to Streamlit
                            window.parent.postMessage({{
                                type: 'streamlit:setComponentValue',
                                question: question,
                                timestamp: new Date().toISOString()
                            }}, '*');
                            
                            // Reset after 3 seconds
                            setTimeout(() => {{
                                if (isListening) {{
                                    fullTranscript = '';
                                    document.getElementById('transcript').innerHTML = '🎤 Ready for next question...';
                                    questionSent = false;
                                }}
                            }}, 3000);
                        }}
                    }}
                }};
                
                recognition.onerror = (event) => {{
                    if (event.error === 'not-allowed') {{
                        document.getElementById('status').innerHTML = '❌ Microphone Access Denied';
                        document.getElementById('status').className = 'stopped';
                        document.getElementById('transcript').innerHTML = 'Please allow microphone access in your browser settings.';
                    }}
                }};
                
                recognition.onend = () => {{
                    if (isListening && !questionSent) {{
                        setTimeout(() => recognition.start(), 100);
                    }} else {{
                        document.getElementById('status').innerHTML = '⏹️ STOPPED';
                        document.getElementById('status').className = 'stopped';
                    }}
                }};
                
                if (isListening) {{
                    recognition.start();
                }}
            }}
            
            init();
        </script>
    </body>
    </html>
    """
    
    result = components.html(speech_html, height=550)
    
    st.info("💡 **Instructions:**\n\n1. Click START LISTENING\n2. Speak your question naturally\n3. Take your time!\n4. Say trigger keyword whenever ready\n5. Answer appears on the right →")

with right_col:
    st.subheader("📝 QUESTIONS & ANSWERS")
    
    # Process new question
    if result and isinstance(result, dict):
        question = result.get('question', '').strip()
        
        if question and not st.session_state.processing:
            st.session_state.processing = True
            
            # Get answers
            with st.spinner("🔍 Searching web..."):
                answers = get_answers(question)
            
            # Add to history
            st.session_state.qa_list.append({
                'question': question,
                'answers': answers,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            st.session_state.processing = False
            st.rerun()
    
    # Display all Q&A (newest first)
    if st.session_state.qa_list:
        for idx, qa in enumerate(reversed(st.session_state.qa_list)):
            qa_num = len(st.session_state.qa_list) - idx
            
            # Question container
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 20px; border-radius: 12px; margin: 15px 0;'>
                <h3>❓ Question {qa_num}</h3>
                <p style='font-size: 18px; line-height: 1.6;'>{qa['question']}</p>
                <small>⏰ {qa['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Answers
            for aidx, ans in enumerate(qa['answers'], 1):
                st.markdown(f"""
                <div style='background: white; padding: 20px; border-radius: 12px; 
                            margin: 10px 0; border-left: 5px solid #4CAF50;'>
                    <h4 style='color: #4CAF50;'>✅ Answer {aidx}</h4>
                    <p style='font-size: 16px; line-height: 1.8; color: #333;'>{ans['answer']}</p>
                    <a href='{ans['source']}' target='_blank' 
                       style='background: #ff9800; color: white; padding: 8px 15px; 
                              border-radius: 20px; text-decoration: none; font-weight: bold;'>
                        🔗 {ans['title']}
                    </a>
                </div>
                """, unsafe_allow_html=True)
            
            # Copy section
            copy_text = f"QUESTION {qa_num}:\n{qa['question']}\n\n"
            for aidx, ans in enumerate(qa['answers'], 1):
                copy_text += f"ANSWER {aidx}:\n{ans['answer']}\n\nSOURCE: {ans['title']}\n{ans['source']}\n\n"
            
            st.text_area(
                f"📋 Copy Q&A #{qa_num}", 
                copy_text, 
                height=250,
                key=f"copy_{qa_num}"
            )
            
            st.markdown("---")
    else:
        st.info("👈 Start speaking on the left. Your Q&A will appear here!")

# Bottom instructions
st.markdown("---")
with st.expander("📖 Detailed Instructions"):
    st.markdown(f"""
    ### How This Works:
    
    **LEFT SIDE (Speech Capture):**
    - Click "START LISTENING" 
    - Speak your question naturally
    - **NO RUSH!** Talk as long as you need
    - Say "{trigger_keyword}" whenever you're ready
    - App waits patiently for the trigger keyword
    
    **RIGHT SIDE (Answers):**
    - Question appears immediately after trigger
    - Answers load from Wikipedia & DuckDuckGo
    - **STAYS VISIBLE** - never disappears
    - All Q&As stack up (newest at top)
    - Copy button for each Q&A
    
    ### Example Flow:
    ```
    You: "What is machine learning?"
    (think for 5 seconds...)
    You: "I understood"
    
    RIGHT SIDE: Question appears → Searching → Answers display → STAYS THERE
    
    You: "What is Python?"
    (wait 10 seconds if you want...)
    You: "I understood"
    
    RIGHT SIDE: New Q&A appears ABOVE the previous one
    ```
    
    ### Tips:
    - ✅ Take your time between question and trigger
    - ✅ All Q&As stay visible permanently
    - ✅ Scroll to see older Q&As
    - ✅ Each Q&A has its own copy button
    - ✅ Click source links to read full articles
    """)

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
import requests
import re
import json
import time

st.set_page_config(
    page_title="AI Interview Assistant",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 60px;
        font-weight: 700;
        font-size: 18px;
        transition: all 0.3s;
    }
    .status-listening {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        animation: pulse 2s infinite;
        margin: 20px 0;
    }
    .status-stopped {
        background: #f8d7da;
        color: #721c24;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    .question-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .answer-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #4CAF50;
    }
    .source-tag {
        display: inline-block;
        background: #ff9800;
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 14px;
        margin: 5px;
        text-decoration: none;
        font-weight: 600;
    }
    .source-tag:hover {
        background: #f57c00;
    }
    </style>
""", unsafe_allow_html=True)

# Session state
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False
if 'last_question_hash' not in st.session_state:
    st.session_state.last_question_hash = None

def clean_repeated_words(text):
    """Remove consecutive duplicate words"""
    words = text.split()
    result = []
    prev = ""
    for word in words:
        if word.lower() != prev.lower() or len(word) < 3:
            result.append(word)
        prev = word
    return " ".join(result)

def search_wikipedia(query):
    """Search Wikipedia API"""
    try:
        url = "https://en.wikipedia.org/w/api.php"
        
        # Search for page
        search_params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'utf8': 1,
            'srlimit': 3
        }
        
        search_resp = requests.get(url, params=search_params, timeout=8)
        search_data = search_resp.json()
        
        results = []
        search_results = search_data.get('query', {}).get('search', [])
        
        for item in search_results[:2]:
            title = item['title']
            
            # Get page extract
            extract_params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'exintro': 1,
                'explaintext': 1
            }
            
            extract_resp = requests.get(url, params=extract_params, timeout=8)
            extract_data = extract_resp.json()
            
            pages = extract_data.get('query', {}).get('pages', {})
            if pages:
                page = list(pages.values())[0]
                extract = page.get('extract', '')
                
                if extract:
                    # Get first 3-4 sentences
                    sentences = re.split(r'(?<=[.!?])\s+', extract)[:4]
                    answer_text = ' '.join(sentences)
                    
                    results.append({
                        'answer': answer_text,
                        'source': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        'title': f'Wikipedia: {title}'
                    })
        
        return results
    except Exception as e:
        return []

def search_duckduckgo(query):
    """Search DuckDuckGo API"""
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1
        }
        
        resp = requests.get(url, params=params, timeout=8)
        data = resp.json()
        
        results = []
        
        if data.get('Abstract'):
            results.append({
                'answer': data['Abstract'],
                'source': data.get('AbstractURL', 'https://duckduckgo.com'),
                'title': data.get('Heading', 'DuckDuckGo Answer')
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
    """Get answers from multiple sources"""
    all_answers = []
    
    # Wikipedia
    wiki_results = search_wikipedia(question)
    all_answers.extend(wiki_results)
    
    # DuckDuckGo
    ddg_results = search_duckduckgo(question)
    all_answers.extend(ddg_results)
    
    if not all_answers:
        all_answers.append({
            'answer': f'No direct answer found for "{question}". This might be a very specific or new topic.',
            'source': f'https://www.google.com/search?q={question.replace(" ", "+")}',
            'title': 'Try Google Search'
        })
    
    return all_answers

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    trigger_keyword = st.text_input(
        "Trigger Keyword",
        value="I understood",
        help="Say this to trigger answer search"
    ).lower().strip()
    
    language = st.selectbox(
        "Language",
        ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "hi-IN"],
        index=0
    )
    
    max_answers = st.slider("Max Answers", 1, 5, 3)
    
    st.markdown("---")
    st.metric("📊 Questions Asked", len(st.session_state.qa_history))
    
    if st.button("🗑️ Clear History"):
        st.session_state.qa_history = []
        st.rerun()

# Main UI
st.title("🎤 AI Interview Assistant")
st.markdown("### Automatic Speech-to-Text with Web Search")

# Control buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ START INTERVIEW", type="primary", use_container_width=True):
        st.session_state.is_listening = True
        st.rerun()

with col2:
    if st.button("⏹️ STOP INTERVIEW", use_container_width=True):
        st.session_state.is_listening = False
        st.rerun()

st.markdown("---")

# Speech recognition component
speech_component = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            padding: 20px;
            background: #f5f7fa;
            margin: 0;
        }}
        #status {{
            font-size: 20px;
            font-weight: bold;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }}
        .listening {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .stopped {{
            background: #f8d7da;
            color: #721c24;
        }}
        #transcript {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            min-height: 100px;
            font-size: 18px;
            line-height: 1.6;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .interim {{ color: #999; font-style: italic; }}
        .detected {{ color: #4CAF50; font-weight: bold; }}
    </style>
</head>
<body>
    <div id="status">Initializing...</div>
    <div id="transcript">Waiting to start...</div>
    
    <script>
        let recognition;
        let isListening = {json.dumps(st.session_state.is_listening)};
        let finalTranscript = '';
        const triggerWord = '{trigger_keyword}';
        const lang = '{language}';
        let questionProcessed = false;
        
        function cleanRepeated(text) {{
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
                document.getElementById('status').innerHTML = '❌ Speech Recognition not supported. Use Chrome/Edge/Safari.';
                document.getElementById('status').className = 'stopped';
                return;
            }}
            
            recognition = new webkitSpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = lang;
            
            recognition.onstart = () => {{
                document.getElementById('status').innerHTML = '🎙️ LISTENING - Ask your question then say "' + triggerWord + '"';
                document.getElementById('status').className = 'listening';
                document.getElementById('transcript').innerHTML = '🎤 Listening...';
                questionProcessed = false;
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
                
                const cleanFinal = cleanRepeated(finalTranscript);
                const cleanInterim = cleanRepeated(interim);
                
                document.getElementById('transcript').innerHTML = 
                    cleanFinal + '<span class="interim">' + cleanInterim + '</span>';
                
                // Check for trigger
                const fullText = (cleanFinal + ' ' + cleanInterim).toLowerCase();
                if (fullText.includes(triggerWord) && !questionProcessed) {{
                    const idx = cleanFinal.toLowerCase().indexOf(triggerWord);
                    if (idx > 0) {{
                        const question = cleanFinal.substring(0, idx).trim();
                        
                        if (question.length > 3) {{
                            questionProcessed = true;
                            
                            document.getElementById('transcript').innerHTML = 
                                '<span class="detected">✅ Question: "' + question + '"</span><br>🔍 Searching web...';
                            
                            // Send to Streamlit
                            const data = {{
                                question: question,
                                timestamp: new Date().toISOString()
                            }};
                            
                            window.parent.postMessage({{
                                type: 'streamlit:setComponentValue',
                                data: data
                            }}, '*');
                            
                            // Reset for next
                            finalTranscript = '';
                            setTimeout(() => {{
                                if (isListening) {{
                                    document.getElementById('transcript').innerHTML = '🎤 Ready for next question...';
                                    questionProcessed = false;
                                }}
                            }}, 3000);
                        }}
                    }}
                }}
            }};
            
            recognition.onerror = (event) => {{
                if (event.error === 'not-allowed') {{
                    document.getElementById('status').innerHTML = '❌ Microphone access denied';
                    document.getElementById('status').className = 'stopped';
                }}
            }};
            
            recognition.onend = () => {{
                if (isListening && !questionProcessed) {{
                    setTimeout(() => recognition.start(), 100);
                }} else {{
                    document.getElementById('status').innerHTML = '⏸️ STOPPED';
                    document.getElementById('status').className = 'stopped';
                }}
            }};
            
            if (isListening) {{
                recognition.start();
            }} else {{
                document.getElementById('status').innerHTML = '⏸️ Click START INTERVIEW to begin';
                document.getElementById('status').className = 'stopped';
            }}
        }}
        
        init();
    </script>
</body>
</html>
"""

# Display speech component
result = components.html(speech_component, height=200)

# Process question automatically
if result and isinstance(result, dict):
    data = result.get('data', {})
    if data and isinstance(data, dict):
        question = data.get('question', '').strip()
        question_hash = hash(question)
        
        # Avoid duplicate processing
        if question and question_hash != st.session_state.last_question_hash:
            st.session_state.last_question_hash = question_hash
            
            # Display question
            st.markdown(
                f'<div class="question-card">'
                f'<h2>❓ YOUR QUESTION:</h2>'
                f'<h3>{question}</h3>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Search and display answers
            with st.spinner("🔍 Searching Wikipedia and DuckDuckGo..."):
                answers = get_answers(question)[:max_answers]
            
            st.markdown("### ✅ ANSWERS:")
            
            for idx, ans in enumerate(answers, 1):
                st.markdown(
                    f'<div class="answer-card">'
                    f'<h4>Answer {idx}:</h4>'
                    f'<p style="font-size: 16px; line-height: 1.8; color: #333;">{ans["answer"]}</p>'
                    f'<a href="{ans["source"]}" target="_blank" class="source-tag">🔗 {ans["title"]}</a>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Save to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.qa_history.append({
                'question': question,
                'answers': answers,
                'timestamp': timestamp
            })
            
            st.success("✅ Saved to history!")
            time.sleep(1)
            st.rerun()

# History section
if st.session_state.qa_history:
    st.markdown("---")
    st.markdown("## 📚 Interview History")
    
    for idx, qa in enumerate(reversed(st.session_state.qa_history), 1):
        with st.expander(f"Q{len(st.session_state.qa_history)-idx+1}: {qa['question'][:60]}... ({qa['timestamp']})"):
            st.markdown(f"**Question:** {qa['question']}")
            st.markdown("**Answers:**")
            
            for aidx, ans in enumerate(qa['answers'], 1):
                st.write(f"{aidx}. {ans['answer']}")
                st.markdown(f"   🔗 [{ans['title']}]({ans['source']})")
            
            # Copy text
            copy_text = f"Q: {qa['question']}\n\n"
            for aidx, ans in enumerate(qa['answers'], 1):
                copy_text += f"A{aidx}: {ans['answer']}\nSource: {ans['source']}\n\n"
            
            st.text_area("Copy", copy_text, height=100, key=f"copy_{idx}")

# Instructions
with st.expander("ℹ️ Instructions"):
    st.markdown("""
    ### How to Use:
    
    1. Click **"▶️ START INTERVIEW"**
    2. Allow microphone access when prompted
    3. **Speak your question**: "What is Python?"
    4. **Say trigger word**: "I understood"
    5. **Automatic**: Question detected → Web search → Answers displayed!
    
    ### No typing, no clicking needed!
    
    - Questions and answers appear automatically
    - All saved in history below
    - Click stop when done
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>💡 Fully Automated Speech Recognition + Web Search</div>", unsafe_allow_html=True)

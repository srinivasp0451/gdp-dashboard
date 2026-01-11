import streamlit as st
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re

# Page configuration
st.set_page_config(
    page_title="AI Interview Assistant",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s;
    }
    .question-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .answer-display {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .source-box {
        background: #fff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .source-link {
        color: #1976D2;
        text-decoration: none;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'transcript_buffer' not in st.session_state:
    st.session_state.transcript_buffer = ""

def clean_text(text):
    """Remove repeated words"""
    words = text.split()
    result = []
    prev = ""
    for word in words:
        if word.lower() != prev.lower() or len(word) < 3:
            result.append(word)
        prev = word
    return " ".join(result)

def search_wikipedia(query):
    """Search Wikipedia"""
    try:
        # Wikipedia API search
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'utf8': 1,
            'srlimit': 1
        }
        
        search_response = requests.get(search_url, params=search_params, timeout=10)
        search_data = search_response.json()
        
        if not search_data.get('query', {}).get('search'):
            return None
        
        # Get the page title
        page_title = search_data['query']['search'][0]['title']
        
        # Get page content
        content_params = {
            'action': 'query',
            'format': 'json',
            'titles': page_title,
            'prop': 'extracts',
            'exintro': 1,
            'explaintext': 1
        }
        
        content_response = requests.get(search_url, params=content_params, timeout=10)
        content_data = content_response.json()
        
        pages = content_data.get('query', {}).get('pages', {})
        if pages:
            page = list(pages.values())[0]
            extract = page.get('extract', '')
            
            # Get first 5 sentences
            sentences = re.split(r'(?<=[.!?])\s+', extract)[:5]
            answer = ' '.join(sentences)
            
            url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
            
            return {
                'answer': answer,
                'source': url,
                'title': f'Wikipedia: {page_title}'
            }
        
        return None
    except Exception as e:
        st.error(f"Wikipedia search error: {str(e)}")
        return None

def search_duckduckgo(query):
    """Search DuckDuckGo Instant Answer API"""
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
                'source': data.get('AbstractURL', 'https://duckduckgo.com'),
                'title': data.get('Heading', 'DuckDuckGo')
            })
        
        # Get related topics
        for topic in data.get('RelatedTopics', [])[:2]:
            if isinstance(topic, dict) and topic.get('Text'):
                results.append({
                    'answer': topic['Text'],
                    'source': topic.get('FirstURL', 'https://duckduckgo.com'),
                    'title': 'DuckDuckGo Related'
                })
        
        return results if results else None
    except Exception as e:
        st.error(f"DuckDuckGo search error: {str(e)}")
        return None

def get_web_answers(question):
    """Get answers from multiple sources"""
    answers = []
    
    with st.spinner("🔍 Searching Wikipedia..."):
        wiki_result = search_wikipedia(question)
        if wiki_result:
            answers.append(wiki_result)
    
    with st.spinner("🔍 Searching DuckDuckGo..."):
        ddg_results = search_duckduckgo(question)
        if ddg_results:
            answers.extend(ddg_results)
    
    if not answers:
        answers.append({
            'answer': f'No results found for "{question}". Try rephrasing or search manually.',
            'source': f'https://www.google.com/search?q={query.replace(" ", "+")}',
            'title': 'Search Google'
        })
    
    return answers

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    trigger_keyword = st.text_input(
        "🎯 Trigger Keyword",
        value="I understood",
        help="Say this after your question"
    ).lower().strip()
    
    max_sources = st.slider("📊 Max Sources", 1, 5, 3)
    
    st.markdown("---")
    st.metric("📚 Total Q&A", len(st.session_state.qa_history))
    
    if st.button("🗑️ Clear All History"):
        st.session_state.qa_history = []
        st.session_state.transcript_buffer = ""
        st.rerun()

# Main UI
st.title("🎤 Interview Assistant with Live Web Search")
st.markdown("### Speak your question, get instant answers from the web!")

# Live Speech Recognition Area
st.markdown("---")
st.subheader("🎙️ Speech Input")

col1, col2 = st.columns([2, 1])

with col1:
    # Live transcript input (simulating speech-to-text)
    transcript = st.text_area(
        "Live Transcript (Type or use speech):",
        value=st.session_state.transcript_buffer,
        height=100,
        placeholder=f"Your speech appears here... say '{trigger_keyword}' to search",
        key="transcript_input",
        help="In a real implementation, this would capture live speech. For now, type your question and the trigger word."
    )
    st.session_state.transcript_buffer = transcript

with col2:
    st.markdown("### 🎯 Quick Actions")
    
    if st.button("🔍 Search Now", type="primary", use_container_width=True):
        if trigger_keyword in transcript.lower():
            # Extract question
            parts = transcript.lower().split(trigger_keyword)
            question = clean_text(parts[0].strip())
            
            if len(question) > 3:
                st.session_state.current_question = question
                st.rerun()
        else:
            st.warning(f"Please add '{trigger_keyword}' after your question!")
    
    if st.button("🔄 Clear Input", use_container_width=True):
        st.session_state.transcript_buffer = ""
        st.rerun()

# Speech Recognition Component (Chrome only)
st.markdown("---")
st.info("💡 **Using Chrome?** Click below to use real speech recognition!")

speech_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ margin: 0; padding: 20px; font-family: Arial; background: #f5f5f5; }}
        .container {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        button {{ 
            padding: 15px 30px; 
            font-size: 16px; 
            border: none; 
            border-radius: 8px; 
            cursor: pointer; 
            margin: 5px;
            font-weight: 600;
        }}
        .start-btn {{ background: #4CAF50; color: white; }}
        .stop-btn {{ background: #f44336; color: white; }}
        #output {{ 
            margin-top: 20px; 
            padding: 15px; 
            background: #e3f2fd; 
            border-radius: 8px;
            min-height: 100px;
            font-size: 16px;
        }}
        .keyword {{ color: #f44336; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <button class="start-btn" onclick="startRecognition()">🎤 Start Speaking</button>
        <button class="stop-btn" onclick="stopRecognition()">⏹️ Stop</button>
        <div id="output">Click "Start Speaking" to begin...</div>
    </div>
    
    <script>
        let recognition;
        let finalTranscript = '';
        const triggerWord = '{trigger_keyword}';
        
        function startRecognition() {{
            if (!('webkitSpeechRecognition' in window)) {{
                document.getElementById('output').innerHTML = 
                    '❌ Speech recognition not supported. Please use Chrome, Edge, or Safari.';
                return;
            }}
            
            recognition = new webkitSpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
            
            recognition.onstart = () => {{
                document.getElementById('output').innerHTML = 
                    '🎙️ <strong>Listening...</strong> Ask your question then say "<span class="keyword">' + triggerWord + '</span>"';
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
                
                const fullText = finalTranscript + interim;
                document.getElementById('output').innerHTML = 
                    '<strong>You said:</strong><br>' + fullText;
                
                // Auto-detect trigger word
                if (fullText.toLowerCase().includes(triggerWord.toLowerCase())) {{
                    const question = fullText.toLowerCase().split(triggerWord.toLowerCase())[0].trim();
                    document.getElementById('output').innerHTML = 
                        '✅ <strong>Question detected!</strong><br>' + question + 
                        '<br><br><em>Copy this question to the transcript box above and click "Search Now"</em>';
                    
                    // Copy to clipboard
                    navigator.clipboard.writeText(question);
                    
                    stopRecognition();
                }}
            }};
            
            recognition.onerror = (event) => {{
                document.getElementById('output').innerHTML = 
                    '❌ Error: ' + event.error;
            }};
            
            recognition.start();
        }}
        
        function stopRecognition() {{
            if (recognition) {{
                recognition.stop();
                document.getElementById('output').innerHTML += '<br><br>⏹️ <em>Stopped listening</em>';
            }}
        }}
    </script>
</body>
</html>
"""

components.html(speech_html, height=250)

# Process and display results
if 'current_question' in st.session_state and st.session_state.current_question:
    question = st.session_state.current_question
    
    st.markdown("---")
    st.markdown("## 🔍 Search Results")
    
    # Display question
    st.markdown(
        f'<div class="question-display">'
        f'<h2>❓ YOUR QUESTION:</h2>'
        f'<h3>{question}</h3>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # Get answers
    answers = get_web_answers(question)[:max_sources]
    
    # Display answers
    st.markdown("### ✅ ANSWERS FROM WEB:")
    
    for idx, ans in enumerate(answers, 1):
        st.markdown(
            f'<div class="answer-display">'
            f'<h3>📝 Answer {idx}:</h3>'
            f'<p style="font-size: 18px; line-height: 1.8;">{ans["answer"]}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        st.markdown(
            f'<div class="source-box">'
            f'🔗 <strong>Source:</strong> '
            f'<a href="{ans["source"]}" target="_blank" class="source-link">{ans["title"]}</a><br>'
            f'<small style="color: #666;">{ans["source"]}</small>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    # Combined text for copying
    combined = f"QUESTION:\n{question}\n\n"
    for idx, ans in enumerate(answers, 1):
        combined += f"ANSWER {idx}:\n{ans['answer']}\n\nSOURCE: {ans['title']}\n{ans['source']}\n\n"
    
    st.text_area("📋 Copy All (Question + Answers + Sources)", value=combined, height=250)
    
    # Save to history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.qa_history.append({
        'question': question,
        'answers': answers,
        'timestamp': timestamp
    })
    
    # Clear for next question
    st.session_state.transcript_buffer = ""
    del st.session_state.current_question
    
    st.success("✅ Saved to history! Ready for next question.")

# History section
if st.session_state.qa_history:
    st.markdown("---")
    st.markdown("## 📚 Interview History")
    
    for idx, qa in enumerate(reversed(st.session_state.qa_history), 1):
        with st.expander(
            f"Q{len(st.session_state.qa_history) - idx + 1}: {qa['question'][:70]}... ({qa['timestamp']})",
            expanded=False
        ):
            st.markdown(f"**❓ Question:** {qa['question']}")
            
            for ans_idx, ans in enumerate(qa['answers'], 1):
                st.markdown(f"**📝 Answer {ans_idx}:**")
                st.write(ans['answer'])
                st.markdown(f"🔗 [{ans['title']}]({ans['source']})")
            
            # Copy option
            combined = f"Q: {qa['question']}\n\n"
            for ans_idx, ans in enumerate(qa['answers'], 1):
                combined += f"A{ans_idx}: {ans['answer']}\nSource: {ans['source']}\n\n"
            
            st.text_area("Copy", combined, height=150, key=f"hist_{idx}")

# Instructions
with st.expander("📖 Instructions"):
    st.markdown("""
    ### How to Use:
    
    **Option 1: Type & Search** (Works everywhere)
    1. Type your question in the transcript box
    2. Add your trigger keyword (e.g., "What is AI I understood")
    3. Click "🔍 Search Now"
    4. See question + answers + sources!
    
    **Option 2: Speech Recognition** (Chrome/Edge/Safari only)
    1. Click "🎤 Start Speaking" in the blue box
    2. Allow microphone access
    3. Speak: "What is machine learning I understood"
    4. Question auto-copies - paste it and click "Search Now"
    
    ### Example:
    ```
    Type or say: "Explain polymorphism in Python I understood"
    
    Results:
    ❓ YOUR QUESTION: Explain polymorphism in Python
    
    ✅ ANSWERS FROM WEB:
    📝 Answer 1: [Wikipedia explanation with source link]
    📝 Answer 2: [DuckDuckGo result with source link]
    ```
    """)

import streamlit.components.v1 as components

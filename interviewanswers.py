import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
import requests
import re
import json

st.set_page_config(page_title="Interview Assistant", page_icon="🎤", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; height: 70px; font-size: 20px; font-weight: bold; border-radius: 12px; }
    </style>
""", unsafe_allow_html=True)

# Session state
if 'qa_list' not in st.session_state:
    st.session_state.qa_list = []
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False
if 'pending_question' not in st.session_state:
    st.session_state.pending_question = None
if 'component_key' not in st.session_state:
    st.session_state.component_key = 0

def clean_repeated_words(text):
    """Remove consecutive duplicate words"""
    if not text:
        return text
    words = text.split()
    result = []
    prev_word = ""
    for word in words:
        clean_word = word.lower().strip()
        if clean_word != prev_word or len(clean_word) < 3:
            result.append(word)
            prev_word = clean_word
    return " ".join(result)

def search_wikipedia(query):
    """Search Wikipedia"""
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
    except Exception as e:
        return []

def search_duckduckgo(query):
    """Search DuckDuckGo"""
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
    """Get answers from multiple sources"""
    all_answers = []
    
    wiki_results = search_wikipedia(question)
    all_answers.extend(wiki_results)
    
    ddg_results = search_duckduckgo(question)
    all_answers.extend(ddg_results)
    
    if not all_answers:
        all_answers.append({
            'answer': f'No specific answer found for "{question}". Try rephrasing your question or search manually.',
            'source': f'https://www.google.com/search?q={question.replace(" ", "+")}',
            'title': 'Search on Google'
        })
    
    return all_answers

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    trigger_keyword = st.text_input(
        "Trigger Keyword",
        value="I understood",
        help="Say this to trigger search - NO RUSH!"
    ).lower().strip()
    
    st.info(f"💡 Say '{trigger_keyword}' ANYTIME after your question!")
    
    st.metric("Total Questions", len(st.session_state.qa_list))
    
    if st.button("🗑️ Clear All"):
        st.session_state.qa_list = []
        st.rerun()

# Title
st.title("🎤 Interview Assistant - Speech to Answer")

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

# Two column layout
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("🎙️ SPEECH INPUT")
    
    # Speech component
    speech_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial; padding: 10px; background: #f0f2f6; margin: 0; }}
            #status {{ font-size: 22px; font-weight: bold; padding: 25px; border-radius: 12px; text-align: center; margin-bottom: 20px; }}
            .listening {{ background: #4CAF50; color: white; animation: pulse 2s infinite; }}
            .stopped {{ background: #f44336; color: white; }}
            @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} }}
            #transcript {{ 
                background: white; 
                padding: 25px; 
                border-radius: 12px; 
                min-height: 300px; 
                max-height: 500px; 
                overflow-y: auto; 
                font-size: 20px; 
                line-height: 2;
                border: 3px solid #2196F3;
                word-wrap: break-word;
            }}
            .final-text {{ color: #000; font-weight: 500; }}
            .interim-text {{ color: #999; font-style: italic; }}
            .captured {{ background: #e8f5e9; padding: 15px; border-radius: 8px; color: #2e7d32; font-weight: bold; margin: 10px 0; border-left: 5px solid #4CAF50; }}
        </style>
    </head>
    <body>
        <div id="status">Click START</div>
        <div id="transcript">Waiting to start...</div>
        
        <script>
            let recognition;
            let isListening = {json.dumps(st.session_state.is_listening)};
            let allFinalText = '';
            const triggerWord = '{trigger_keyword}';
            let alreadyProcessed = false;
            
            function removeDuplicates(text) {{
                if (!text) return '';
                const words = text.split(' ');
                const cleaned = [];
                let prevWord = '';
                
                for (let i = 0; i < words.length; i++) {{
                    const currentWord = words[i].toLowerCase().trim();
                    if (currentWord !== prevWord || currentWord.length < 3) {{
                        cleaned.push(words[i]);
                    }}
                    prevWord = currentWord;
                }}
                
                return cleaned.join(' ');
            }}
            
            function init() {{
                if (!('webkitSpeechRecognition' in window)) {{
                    document.getElementById('status').innerHTML = '❌ Not Supported';
                    document.getElementById('status').className = 'stopped';
                    document.getElementById('transcript').innerHTML = 'Speech recognition not supported. Use Chrome, Edge, or Safari.';
                    return;
                }}
                
                recognition = new webkitSpeechRecognition();
                recognition.continuous = true;
                recognition.interimResults = true;
                recognition.lang = 'en-US';
                recognition.maxAlternatives = 1;
                
                recognition.onstart = () => {{
                    document.getElementById('status').innerHTML = '🎙️ LISTENING - Speak your question, then say "' + triggerWord + '"';
                    document.getElementById('status').className = 'listening';
                    alreadyProcessed = false;
                    allFinalText = '';
                }};
                
                recognition.onresult = (event) => {{
                    let interimText = '';
                    let newFinalText = '';
                    
                    // Collect all final results
                    for (let i = 0; i < event.results.length; i++) {{
                        if (event.results[i].isFinal) {{
                            newFinalText += event.results[i][0].transcript + ' ';
                        }} else {{
                            interimText += event.results[i][0].transcript;
                        }}
                    }}
                    
                    // Update accumulated final text
                    if (newFinalText) {{
                        allFinalText = newFinalText;
                    }}
                    
                    // Clean both texts
                    const cleanedFinal = removeDuplicates(allFinalText);
                    const cleanedInterim = removeDuplicates(interimText);
                    
                    // Display EVERYTHING - never clear
                    document.getElementById('transcript').innerHTML = 
                        '<span class="final-text">' + cleanedFinal + '</span> ' +
                        '<span class="interim-text">' + cleanedInterim + '</span>';
                    
                    // Check for trigger ONLY in final text
                    const lowerFinal = cleanedFinal.toLowerCase();
                    if (lowerFinal.includes(triggerWord) && !alreadyProcessed) {{
                        alreadyProcessed = true;
                        
                        const triggerIndex = lowerFinal.indexOf(triggerWord);
                        const question = cleanedFinal.substring(0, triggerIndex).trim();
                        
                        if (question.length > 5) {{
                            document.getElementById('transcript').innerHTML = 
                                '<div class="captured">✅ QUESTION CAPTURED!</div>' +
                                '<div style="font-size: 18px; margin: 15px 0;"><strong>Question:</strong> ' + question + '</div>' +
                                '<div style="color: #ff9800; font-weight: bold;">🔍 Searching for answers...</div>';
                            
                            // Send to Streamlit using query params hack
                            const question_encoded = encodeURIComponent(question);
                            const timestamp_encoded = new Date().getTime();
                            
                            // Try multiple methods to communicate
                            window.parent.postMessage({{
                                type: 'streamlit:setComponentValue',
                                key: 'speech_result',
                                value: {{
                                    question: question,
                                    timestamp: timestamp_encoded
                                }}
                            }}, '*');
                            
                            // Also try storing in session storage
                            try {{
                                sessionStorage.setItem('pending_question', question);
                                sessionStorage.setItem('question_timestamp', timestamp_encoded);
                            }} catch(e) {{}}
                            
                            // Force parent refresh
                            window.parent.postMessage({{
                                type: 'streamlit:rerun'
                            }}, '*');
                            
                            // Reset after delay
                            setTimeout(() => {{
                                allFinalText = '';
                                alreadyProcessed = false;
                                document.getElementById('transcript').innerHTML = 
                                    '<div style="color: #4CAF50; font-weight: bold; font-size: 20px;">✅ Answer added to the right →</div>' +
                                    '<div style="margin-top: 15px;">🎤 Ready for your next question...</div>';
                            }}, 3000);
                        }}
                    }}
                }};
                
                recognition.onerror = (event) => {{
                    if (event.error === 'not-allowed') {{
                        document.getElementById('status').innerHTML = '❌ Microphone Denied';
                        document.getElementById('status').className = 'stopped';
                        document.getElementById('transcript').innerHTML = 'Please allow microphone access in your browser.';
                    }}
                }};
                
                recognition.onend = () => {{
                    if (isListening) {{
                        // Keep restarting while listening
                        setTimeout(() => {{
                            try {{ recognition.start(); }} catch(e) {{ }}
                        }}, 100);
                    }} else {{
                        document.getElementById('status').innerHTML = '⏹️ STOPPED';
                        document.getElementById('status').className = 'stopped';
                    }}
                }};
                
                if (isListening) {{
                    try {{ recognition.start(); }} catch(e) {{ }}
                }}
            }}
            
            init();
        </script>
    </body>
    </html>
    """
    
    # Add JavaScript to check session storage
    check_storage_html = """
    <script>
        const question = sessionStorage.getItem('pending_question');
        const timestamp = sessionStorage.getItem('question_timestamp');
        
        if (question && timestamp) {
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: {
                    question: question,
                    timestamp: timestamp
                }
            }, '*');
            
            // Clear after sending
            sessionStorage.removeItem('pending_question');
            sessionStorage.removeItem('question_timestamp');
        }
    </script>
    """
    
    check_result = components.html(check_storage_html, height=0)
    
    result = components.html(speech_html, height=650)
    
    # Debug: Show what we're receiving
    st.write("DEBUG - check_result type:", type(check_result))
    st.write("DEBUG - result type:", type(result))
    if check_result and isinstance(check_result, dict):
        st.write("DEBUG - check_result:", check_result)
    if result and isinstance(result, dict):
        st.write("DEBUG - result:", result)
    
    st.info("**💡 How to use:**\n\n1. Click START LISTENING\n2. Speak your question naturally\n3. Take your time - NO RUSH\n4. Say trigger keyword when ready\n5. Answers appear on right →")

with right_col:
    st.subheader("📝 QUESTIONS & ANSWERS")
    
    # Try to get data from both components
    question_to_process = None
    
    # Check storage component
    if check_result and isinstance(check_result, dict):
        q = check_result.get('question', '').strip()
        if q:
            question_to_process = q
            st.success(f"✅ Got question from storage: {q}")
    
    # Check speech component
    if result and isinstance(result, dict):
        q = result.get('question', '').strip()
        if q:
            question_to_process = q
            st.success(f"✅ Got question from speech: {q}")
    
    # Process if we have a question
    if question_to_process:
        # Check if not duplicate
        if not st.session_state.qa_list or st.session_state.qa_list[-1]['question'] != question_to_process:
            st.info("🔍 Searching for answers...")
            
            # Get answers
            answers = get_answers(question_to_process)
            
            # Add to list
            st.session_state.qa_list.append({
                'question': question_to_process,
                'answers': answers,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            st.success(f"✅ Added! Total Q&As: {len(st.session_state.qa_list)}")
            st.rerun()
    
    st.markdown("---")
    
    # Display all Q&A
    if st.session_state.qa_list:
        for idx in range(len(st.session_state.qa_list) - 1, -1, -1):
            qa = st.session_state.qa_list[idx]
            qa_num = idx + 1
            
            # Question
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 25px; border-radius: 15px; margin: 20px 0; box-shadow: 0 5px 15px rgba(0,0,0,0.3);'>
                <h2 style='margin: 0 0 10px 0;'>❓ Question {qa_num}</h2>
                <p style='font-size: 20px; line-height: 1.8; margin: 10px 0;'>{qa['question']}</p>
                <small style='opacity: 0.9;'>⏰ {qa['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Answers
            for aidx, ans in enumerate(qa['answers'], 1):
                st.markdown(f"""
                <div style='background: white; padding: 25px; border-radius: 15px; 
                            margin: 15px 0; border-left: 6px solid #4CAF50; box-shadow: 0 3px 10px rgba(0,0,0,0.1);'>
                    <h3 style='color: #4CAF50; margin-top: 0;'>✅ Answer {aidx}</h3>
                    <p style='font-size: 17px; line-height: 1.9; color: #333; margin: 15px 0;'>{ans['answer']}</p>
                    <a href='{ans['source']}' target='_blank' 
                       style='display: inline-block; background: #ff9800; color: white; padding: 10px 20px; 
                              border-radius: 25px; text-decoration: none; font-weight: bold; margin-top: 10px;'>
                        🔗 {ans['title']}
                    </a>
                </div>
                """, unsafe_allow_html=True)
            
            # Copy area
            copy_text = f"QUESTION {qa_num}:\n{qa['question']}\n\n"
            for aidx, ans in enumerate(qa['answers'], 1):
                copy_text += f"ANSWER {aidx}:\n{ans['answer']}\n\nSOURCE: {ans['title']}\n{ans['source']}\n\n{'='*80}\n\n"
            
            st.text_area(
                f"📋 Copy Q&A #{qa_num}", 
                copy_text, 
                height=280,
                key=f"copy_{qa_num}_{qa['timestamp']}"
            )
            
            st.markdown("<hr style='margin: 30px 0; border: 2px solid #ddd;'>", unsafe_allow_html=True)
    else:
        st.info("👈 Start speaking on the left!\n\nYour questions and answers will appear here and STAY VISIBLE.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; padding: 15px;'>💡 Text NEVER disappears • Continuous listening • All Q&As stay visible</div>", unsafe_allow_html=True)

import streamlit as st
import streamlit.components.v1 as components
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
        
        page_title = search_data['query']['search'][0]['title']
        
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
        st.error(f"Wikipedia error: {str(e)}")
        return None

def search_duckduckgo(query):
    """Search DuckDuckGo"""
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
        
        if data.get('Abstract'):
            results.append({
                'answer': data['Abstract'],
                'source': data.get('AbstractURL', 'https://duckduckgo.com'),
                'title': data.get('Heading', 'DuckDuckGo')
            })
        
        for topic in data.get('RelatedTopics', [])[:2]:
            if isinstance(topic, dict) and topic.get('Text'):
                results.append({
                    'answer': topic['Text'],
                    'source': topic.get('FirstURL', 'https://duckduckgo.com'),
                    'title': 'DuckDuckGo Related'
                })
        
        return results if results else None
    except Exception as e:
        st.error(f"DuckDuckGo error: {str(e)}")
        return None

def get_web_answers(question):
    """Get answers from web"""
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
            'answer': f'No results found for "{question}". Try rephrasing.',
            'source': f'https://www.google.com/search?q={question.replace(" ", "+")}',
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
    
    if st.button("🗑️ Clear All"):
        st.session_state.qa_history = []
        st.session_state.transcript_buffer = ""
        st.rerun()

# Main UI
st.title("🎤 Interview Assistant with Web Search")
st.markdown("### Type your question + trigger word, then click Search!")

st.markdown("---")

# Input area
col1, col2 = st.columns([3, 1])

with col1:
    transcript = st.text_area(
        "📝 Type Your Question:",
        value=st.session_state.transcript_buffer,
        height=120,
        placeholder=f"Example: What is Python {trigger_keyword}",
        key="transcript_input"
    )
    st.session_state.transcript_buffer = transcript

with col2:
    st.markdown("### Actions")
    
    if st.button("🔍 Search Now", type="primary", use_container_width=True):
        if trigger_keyword in transcript.lower():
            parts = transcript.lower().split(trigger_keyword)
            question = clean_text(parts[0].strip())
            
            if len(question) > 3:
                st.session_state.current_question = question
                st.rerun()
            else:
                st.error("Question too short!")
        else:
            st.warning(f"Add '{trigger_keyword}' after question!")
    
    if st.button("🔄 Clear", use_container_width=True):
        st.session_state.transcript_buffer = ""
        st.rerun()

# Process and display results
if 'current_question' in st.session_state and st.session_state.current_question:
    question = st.session_state.current_question
    
    st.markdown("---")
    st.markdown("## 🔍 SEARCH RESULTS")
    
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
    
    # Combined text
    combined = f"QUESTION:\n{question}\n\n"
    for idx, ans in enumerate(answers, 1):
        combined += f"ANSWER {idx}:\n{ans['answer']}\n\nSOURCE: {ans['title']}\n{ans['source']}\n\n"
    
    st.text_area("📋 COPY ALL (Question + Answers + Sources)", value=combined, height=300)
    
    # Save to history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.qa_history.append({
        'question': question,
        'answers': answers,
        'timestamp': timestamp
    })
    
    st.session_state.transcript_buffer = ""
    del st.session_state.current_question
    
    st.success("✅ Saved to history! Ready for next question.")

# History
if st.session_state.qa_history:
    st.markdown("---")
    st.markdown("## 📚 HISTORY")
    
    for idx, qa in enumerate(reversed(st.session_state.qa_history), 1):
        with st.expander(
            f"Q{len(st.session_state.qa_history) - idx + 1}: {qa['question'][:60]}... ({qa['timestamp']})"
        ):
            st.markdown(f"**❓ Question:** {qa['question']}")
            
            for ans_idx, ans in enumerate(qa['answers'], 1):
                st.markdown(f"**📝 Answer {ans_idx}:**")
                st.write(ans['answer'])
                st.markdown(f"🔗 [{ans['title']}]({ans['source']})")
            
            combined = f"Q: {qa['question']}\n\n"
            for ans_idx, ans in enumerate(qa['answers'], 1):
                combined += f"A{ans_idx}: {ans['answer']}\nSource: {ans['source']}\n\n"
            
            st.text_area("Copy", combined, height=120, key=f"h_{idx}")

# Instructions
with st.expander("📖 HOW TO USE"):
    st.markdown("""
    ### Simple Steps:
    
    1. **Type your question** in the text box
    2. **Add trigger keyword** at the end (default: "I understood")
    3. **Click "🔍 Search Now"**
    4. **See results**: Question + Answers from Wikipedia & DuckDuckGo + Source links
    
    ### Example:
    ```
    Type: "What is machine learning I understood"
    Click: Search Now
    
    Get:
    ❓ YOUR QUESTION: What is machine learning
    
    ✅ ANSWERS FROM WEB:
    📝 Answer 1: [Wikipedia explanation]
    🔗 Source: Wikipedia link
    
    📝 Answer 2: [DuckDuckGo result]
    🔗 Source: DuckDuckGo link
    
    📋 Copy box with everything combined
    ```
    
    ### Tips:
    - Always add trigger keyword after your question
    - Questions must be at least 3 words
    - Click source links to read full articles
    - All Q&A saved in History section
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "💡 Searches Wikipedia & DuckDuckGo automatically!"
    "</div>",
    unsafe_allow_html=True
)

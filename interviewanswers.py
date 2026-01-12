import streamlit as st
import requests
import re
from datetime import datetime

st.set_page_config(page_title="Interview Assistant", page_icon="🎤", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .big-title { font-size: 32px; font-weight: bold; color: #1f1f1f; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# Session state
if 'qa_list' not in st.session_state:
    st.session_state.qa_list = []
if 'current_transcript' not in st.session_state:
    st.session_state.current_transcript = ""

def clean_repeated_words(text):
    """Remove consecutive duplicate words"""
    if not text:
        return text
    words = text.split()
    result = []
    prev = ""
    for word in words:
        if word.lower() != prev.lower() or len(word) < 3:
            result.append(word)
            prev = word.lower()
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
    """Get answers from web"""
    all_answers = []
    
    wiki = search_wikipedia(question)
    all_answers.extend(wiki)
    
    ddg = search_duckduckgo(question)
    all_answers.extend(ddg)
    
    if not all_answers:
        all_answers.append({
            'answer': f'No specific answer found for "{question}". Try rephrasing or searching manually.',
            'source': f'https://www.google.com/search?q={question.replace(" ", "+")}',
            'title': 'Search on Google'
        })
    
    return all_answers

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    trigger = st.text_input("Trigger Keyword", value="I understood").lower().strip()
    
    st.info(f"💡 Type or paste: 'Your question {trigger}'")
    
    st.metric("Total Questions", len(st.session_state.qa_list))
    
    if st.button("🗑️ Clear All"):
        st.session_state.qa_list = []
        st.session_state.current_transcript = ""
        st.rerun()

# Main UI
st.markdown('<p class="big-title">🎤 Interview Assistant</p>', unsafe_allow_html=True)

st.info("""
**📝 SIMPLE WORKFLOW:**
1. Use your phone/computer's speech-to-text (dictation) to speak your question
2. Paste it in the box below (it should include your trigger word)
3. Click "Search for Answer"
4. Answer appears below with sources!

**OR** just type: "What is Python I understood" and click Search
""")

# Input section
col1, col2 = st.columns([4, 1])

with col1:
    transcript = st.text_area(
        "📝 Paste or Type Your Question Here:",
        value=st.session_state.current_transcript,
        height=150,
        placeholder=f"Example: What is machine learning {trigger}",
        help="Use your device's speech-to-text to dictate, then paste here"
    )
    st.session_state.current_transcript = transcript

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    
    if st.button("🔍 SEARCH", type="primary", use_container_width=True):
        cleaned = clean_repeated_words(transcript)
        
        if trigger in cleaned.lower():
            parts = cleaned.lower().split(trigger)
            question = parts[0].strip()
            
            if len(question) > 3:
                # Get answers
                with st.spinner("🔍 Searching Wikipedia and DuckDuckGo..."):
                    answers = get_answers(question)
                
                # Add to history
                st.session_state.qa_list.append({
                    'question': question,
                    'answers': answers,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
                # Clear input
                st.session_state.current_transcript = ""
                st.rerun()
            else:
                st.error("❌ Question too short!")
        else:
            st.error(f"❌ Please add '{trigger}' after your question!")
    
    if st.button("🔄 Clear", use_container_width=True):
        st.session_state.current_transcript = ""
        st.rerun()

st.markdown("---")

# Display all Q&A
if st.session_state.qa_list:
    st.markdown("## 📚 QUESTIONS & ANSWERS")
    
    for idx in range(len(st.session_state.qa_list) - 1, -1, -1):
        qa = st.session_state.qa_list[idx]
        qa_num = idx + 1
        
        # Question Box
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 30px; border-radius: 15px; margin: 25px 0; 
                    box-shadow: 0 8px 20px rgba(0,0,0,0.3);'>
            <h1 style='margin: 0 0 15px 0; font-size: 28px;'>❓ QUESTION {qa_num}</h1>
            <p style='font-size: 22px; line-height: 1.9; margin: 15px 0; font-weight: 500;'>{qa['question']}</p>
            <small style='opacity: 0.9; font-size: 16px;'>⏰ {qa['timestamp']}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Answers
        for aidx, ans in enumerate(qa['answers'], 1):
            st.markdown(f"""
            <div style='background: white; padding: 30px; border-radius: 15px; 
                        margin: 20px 0; border-left: 8px solid #4CAF50; 
                        box-shadow: 0 5px 15px rgba(0,0,0,0.15);'>
                <h2 style='color: #4CAF50; margin-top: 0; font-size: 24px;'>✅ ANSWER {aidx}</h2>
                <p style='font-size: 19px; line-height: 2; color: #1f1f1f; margin: 20px 0;'>{ans['answer']}</p>
                <a href='{ans['source']}' target='_blank' 
                   style='display: inline-block; background: #ff9800; color: white; 
                          padding: 12px 25px; border-radius: 30px; text-decoration: none; 
                          font-weight: bold; font-size: 16px; margin-top: 15px;
                          box-shadow: 0 4px 10px rgba(255,152,0,0.3);'>
                    🔗 SOURCE: {ans['title']}
                </a>
            </div>
            """, unsafe_allow_html=True)
        
        # Copy box - LARGE
        copy_text = f"{'='*100}\nQUESTION {qa_num}:\n{'='*100}\n\n{qa['question']}\n\n"
        for aidx, ans in enumerate(qa['answers'], 1):
            copy_text += f"\n{'='*100}\nANSWER {aidx}:\n{'='*100}\n\n{ans['answer']}\n\nSOURCE: {ans['title']}\nURL: {ans['source']}\n\n"
        
        st.text_area(
            f"📋 COPY EVERYTHING - Question {qa_num}",
            copy_text,
            height=350,
            key=f"copy_{qa_num}"
        )
        
        st.markdown("<hr style='border: 3px solid #ddd; margin: 40px 0;'>", unsafe_allow_html=True)
else:
    st.info("👆 Enter your question above and click SEARCH to get started!")

st.markdown("---")

with st.expander("📖 How to Use This App"):
    st.markdown("""
    ### Simple Method:
    
    **On Phone/Computer:**
    1. Use your device's built-in speech-to-text (microphone button on keyboard)
    2. Speak: "What is artificial intelligence I understood"
    3. Copy the transcribed text
    4. Paste in the box above
    5. Click "SEARCH"
    6. Answers appear with sources!
    
    **Or Just Type:**
    1. Type: "What is Python I understood"
    2. Click "SEARCH"
    3. Done!
    
    ### Features:
    - ✅ All Q&As stay visible permanently
    - ✅ Large, readable text (350px copy areas)
    - ✅ Proper question/answer labels
    - ✅ Source links you can click
    - ✅ Clean duplicate words automatically
    - ✅ Searches Wikipedia + DuckDuckGo
    
    ### Tips:
    - Works on ANY device (phone, tablet, computer)
    - Use device's native speech-to-text for best results
    - All answers stay visible - scroll to see older ones
    - Copy boxes are LARGE (350px) for easy reading
    """)

st.markdown("""
<div style='text-align: center; color: #666; padding: 20px; font-size: 16px;'>
    💡 <strong>SIMPLE & WORKING:</strong> Use your device's speech-to-text → Paste → Search → Get Answers!
</div>
""", unsafe_allow_html=True)

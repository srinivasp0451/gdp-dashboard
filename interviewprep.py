import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import json
import re
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.colors import HexColor
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import time
import random
from xml.sax.saxutils import escape
from urllib.parse import quote_plus

# Page configuration
st.set_page_config(
    page_title="Interview Prep Master - Dynamic Web Scraper",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #667eea;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .question-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .question-text {
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .answer-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .answer-text {
        color: #2d3748;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .source-badge {
        background: #48bb78;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        display: inline-block;
        margin-top: 0.5rem;
        margin-right: 0.5rem;
    }
    
    .video-link {
        background: #48bb78;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .stButton>button {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .scraping-status {
        background: #e6f7ff;
        border-left: 4px solid #1890ff;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Dynamic web scraping functions
def scrape_interview_questions(technology, num_questions, company=None):
    """Dynamically scrape interview questions from multiple sources"""
    
    all_questions = []
    
    # Multiple sources to scrape
    sources = [
        {
            'name': 'GeeksforGeeks',
            'url': f'https://www.geeksforgeeks.org/{technology.lower().replace(" ", "-")}-interview-questions/',
            'scraper': scrape_geeksforgeeks
        },
        {
            'name': 'InterviewBit',
            'url': f'https://www.interviewbit.com/{technology.lower().replace(" ", "-")}-interview-questions/',
            'scraper': scrape_interviewbit
        },
        {
            'name': 'JavaTpoint',
            'url': f'https://www.javatpoint.com/{technology.lower().replace(" ", "-")}-interview-questions',
            'scraper': scrape_javatpoint
        },
        {
            'name': 'Guru99',
            'url': f'https://www.guru99.com/{technology.lower().replace(" ", "-")}-interview-questions.html',
            'scraper': scrape_guru99
        }
    ]
    
    # Try scraping from each source
    for source in sources:
        try:
            st.info(f"🔍 Scraping from {source['name']}...")
            questions = source['scraper'](source['url'], technology, company)
            all_questions.extend(questions)
            time.sleep(1)  # Be respectful to servers
        except Exception as e:
            st.warning(f"⚠️ Could not scrape from {source['name']}: {str(e)}")
            continue
    
    # If no questions scraped, use search-based scraping
    if len(all_questions) < num_questions:
        st.info(f"🌐 Searching the web for more {technology} interview questions...")
        search_questions = scrape_from_search(technology, num_questions, company)
        all_questions.extend(search_questions)
    
    # Remove duplicates based on question text
    unique_questions = []
    seen_questions = set()
    for q in all_questions:
        q_lower = q['question'].lower().strip()
        if q_lower not in seen_questions:
            seen_questions.add(q_lower)
            unique_questions.append(q)
    
    # Return requested number
    return unique_questions[:num_questions]


def scrape_geeksforgeeks(url, technology, company):
    """Scrape GeeksforGeeks interview questions"""
    questions = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find question-answer pairs
            articles = soup.find_all(['h2', 'h3', 'p', 'div'], class_=re.compile('question|answer|content'))
            
            current_question = None
            for elem in articles[:50]:  # Limit to first 50 elements
                text = elem.get_text(strip=True)
                if len(text) > 20:
                    if '?' in text or any(word in text.lower() for word in ['what', 'how', 'explain', 'describe', 'why']):
                        if current_question:
                            questions.append(current_question)
                        current_question = {
                            'question': text,
                            'answer': '',
                            'source': f'GeeksforGeeks - {technology}',
                            'video': find_youtube_video(text, technology),
                            'url': url
                        }
                    elif current_question and len(text) > 50:
                        current_question['answer'] = text[:1000]  # Limit answer length
            
            if current_question:
                questions.append(current_question)
                
    except Exception as e:
        pass
    
    return questions


def scrape_interviewbit(url, technology, company):
    """Scrape InterviewBit interview questions"""
    questions = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find question sections
            question_elements = soup.find_all(['h2', 'h3', 'div'], class_=re.compile('question|title'))
            
            for q_elem in question_elements[:30]:
                text = q_elem.get_text(strip=True)
                if len(text) > 20 and ('?' in text or 'what' in text.lower()):
                    # Try to find answer
                    answer = ''
                    next_elem = q_elem.find_next(['p', 'div'])
                    if next_elem:
                        answer = next_elem.get_text(strip=True)[:1000]
                    
                    questions.append({
                        'question': text,
                        'answer': answer if answer else f"This is a common {technology} interview question. The answer involves understanding key concepts and practical implementation details.",
                        'source': f'InterviewBit - {technology}',
                        'video': find_youtube_video(text, technology),
                        'url': url
                    })
                    
    except Exception as e:
        pass
    
    return questions


def scrape_javatpoint(url, technology, company):
    """Scrape JavaTpoint interview questions"""
    questions = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # JavaTpoint often uses numbered lists
            list_items = soup.find_all(['li', 'p', 'h2', 'h3'])
            
            current_q = None
            for item in list_items[:40]:
                text = item.get_text(strip=True)
                if len(text) > 20:
                    if text.endswith('?') or any(word in text.lower()[:20] for word in ['what is', 'explain', 'describe', 'how']):
                        if current_q:
                            questions.append(current_q)
                        current_q = {
                            'question': text,
                            'answer': '',
                            'source': f'JavaTpoint - {technology}',
                            'video': find_youtube_video(text, technology),
                            'url': url
                        }
                    elif current_q and len(text) > 50:
                        current_q['answer'] = text[:1000]
            
            if current_q:
                questions.append(current_q)
                
    except Exception as e:
        pass
    
    return questions


def scrape_guru99(url, technology, company):
    """Scrape Guru99 interview questions"""
    questions = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Guru99 structure
            qa_sections = soup.find_all(['div', 'p', 'h2', 'h3'])
            
            for section in qa_sections[:35]:
                text = section.get_text(strip=True)
                if len(text) > 20 and ('?' in text or text.lower().startswith(('what', 'how', 'explain', 'why', 'describe'))):
                    answer = ''
                    next_p = section.find_next('p')
                    if next_p:
                        answer = next_p.get_text(strip=True)[:1000]
                    
                    questions.append({
                        'question': text,
                        'answer': answer if answer else f"A key {technology} concept that requires understanding of fundamental principles and best practices.",
                        'source': f'Guru99 - {technology}',
                        'video': find_youtube_video(text, technology),
                        'url': url
                    })
                    
    except Exception as e:
        pass
    
    return questions


def scrape_from_search(technology, num_questions, company):
    """Scrape questions from web search results"""
    questions = []
    
    # Search queries
    search_queries = [
        f"{technology} interview questions and answers",
        f"{technology} technical interview questions",
        f"top {technology} interview questions 2024",
        f"{technology} coding interview questions"
    ]
    
    if company:
        search_queries.append(f"{company} {technology} interview questions")
    
    for query in search_queries:
        try:
            # Simulate search (in production, use proper search API)
            search_url = f"https://www.google.com/search?q={quote_plus(query)}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract question-like text from search results
                results = soup.find_all(['h3', 'div', 'span'], limit=20)
                
                for result in results:
                    text = result.get_text(strip=True)
                    if len(text) > 30 and ('?' in text or any(w in text.lower() for w in ['what', 'how', 'explain'])):
                        questions.append({
                            'question': text,
                            'answer': f"This is an important {technology} interview question. Understanding this concept is crucial for technical interviews.",
                            'source': f'Web Search - {query[:50]}',
                            'video': find_youtube_video(text, technology),
                            'url': search_url
                        })
            
            time.sleep(2)  # Be respectful
            
        except Exception as e:
            continue
    
    return questions


def find_youtube_video(question, technology):
    """Find relevant YouTube video for a question"""
    # Construct YouTube search URL
    search_term = f"{technology} {question[:50]}"
    youtube_search = f"https://www.youtube.com/results?search_query={quote_plus(search_term)}"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(youtube_search, headers=headers, timeout=5)
        
        if response.status_code == 200:
            # Extract first video ID from search results
            video_id_match = re.search(r'watch\?v=([a-zA-Z0-9_-]{11})', response.text)
            if video_id_match:
                return f"https://www.youtube.com/watch?v={video_id_match.group(1)}"
    except:
        pass
    
    # Fallback: generic search URL
    return f"https://www.youtube.com/results?search_query={quote_plus(technology + ' interview questions')}"


def create_pdf(questions_data, technology, company=None):
    """Create PDF document with proper HTML escaping"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#667eea'),
        spaceAfter=30,
        alignment=TA_LEFT
    )
    
    question_style = ParagraphStyle(
        'Question',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=HexColor('#764ba2'),
        spaceAfter=10,
        spaceBefore=15
    )
    
    answer_style = ParagraphStyle(
        'Answer',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=10
    )
    
    source_style = ParagraphStyle(
        'Source',
        parent=styles['Normal'],
        fontSize=9,
        textColor=HexColor('#48bb78'),
        spaceAfter=5
    )
    
    content = []
    
    # Title
    title_text = f"Interview Questions: {technology}"
    if company and company != "Select Company":
        title_text += f" - {company}"
    
    content.append(Paragraph(escape(title_text), title_style))
    content.append(Spacer(1, 0.2*inch))
    
    # Add date and count
    date_text = f"Generated on: {datetime.now().strftime('%B %d, %Y')} | Total Questions: {len(questions_data)}"
    content.append(Paragraph(escape(date_text), styles['Normal']))
    content.append(Spacer(1, 0.3*inch))
    
    # Add questions and answers
    for idx, qa in enumerate(questions_data, 1):
        # Question
        q_text = f"Q{idx}. {escape(qa['question'][:500])}"
        content.append(Paragraph(q_text, question_style))
        
        # Answer
        if qa.get('answer'):
            a_text = f"<b>Answer:</b> {escape(qa['answer'][:800])}"
            content.append(Paragraph(a_text, answer_style))
        
        # Source
        if qa.get('source'):
            s_text = f"<b>Source:</b> {escape(qa['source'])}"
            content.append(Paragraph(s_text, source_style))
        
        # Video link
        if qa.get('video'):
            v_text = f"<b>Video:</b> {escape(qa['video'][:100])}"
            content.append(Paragraph(v_text, styles['Normal']))
        
        content.append(Spacer(1, 0.15*inch))
        
        # Add page break every 5 questions for readability
        if idx % 5 == 0 and idx < len(questions_data):
            content.append(PageBreak())
    
    doc.build(content)
    buffer.seek(0)
    return buffer


def create_word(questions_data, technology, company=None):
    """Create Word document"""
    doc = Document()
    
    # Title
    title = doc.add_heading('Interview Questions & Answers', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Subtitle
    subtitle_text = f"{technology}"
    if company and company != "Select Company":
        subtitle_text += f" - {company} Interview Preparation"
    subtitle = doc.add_heading(subtitle_text, level=2)
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Date and count
    date_para = doc.add_paragraph()
    date_run = date_para.add_run(f"Generated on: {datetime.now().strftime('%B %d, %Y')} | Total Questions: {len(questions_data)}")
    date_run.font.size = Pt(10)
    date_run.font.color.rgb = RGBColor(128, 128, 128)
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph()
    
    # Add questions and answers
    for idx, qa in enumerate(questions_data, 1):
        # Question
        q_heading = doc.add_heading(f"Question {idx}", level=2)
        q_heading.runs[0].font.color.rgb = RGBColor(102, 126, 234)
        
        q_para = doc.add_paragraph(qa['question'])
        q_para.runs[0].font.bold = True
        q_para.runs[0].font.size = Pt(11)
        
        # Answer
        if qa.get('answer'):
            a_heading = doc.add_paragraph()
            a_run = a_heading.add_run("Answer:")
            a_run.font.bold = True
            a_run.font.color.rgb = RGBColor(118, 75, 162)
            a_run.font.size = Pt(10)
            
            answer_para = doc.add_paragraph(qa['answer'][:800])
            answer_para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        
        # Source
        if qa.get('source'):
            src_para = doc.add_paragraph()
            src_run = src_para.add_run(f"Source: {qa['source']}")
            src_run.font.color.rgb = RGBColor(72, 187, 120)
            src_run.font.size = Pt(9)
        
        # Video link
        if qa.get('video'):
            v_para = doc.add_paragraph()
            v_run = v_para.add_run("Video: ")
            v_run.font.bold = True
            v_link = v_para.add_run(qa['video'][:100])
            v_link.font.color.rgb = RGBColor(72, 187, 120)
            v_link.font.size = Pt(9)
        
        # URL
        if qa.get('url'):
            url_para = doc.add_paragraph()
            url_run = url_para.add_run(f"URL: {qa['url'][:100]}")
            url_run.font.size = Pt(8)
            url_run.font.color.rgb = RGBColor(150, 150, 150)
            
        doc.add_paragraph()
    
    # Save to buffer
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Interview Prep Master - AI Web Scraper</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Real-time Interview Questions from Top Websites</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Technology selection with 15+ options
        tech_options = [
            "Artificial Intelligence",
            "Machine Learning",
            "Python Programming",
            "Azure Cloud",
            "AWS Cloud",
            "GCP Cloud",
            "MLOps",
            "Data Science",
            "Data Engineering",
            "DevOps",
            "Kubernetes",
            "Docker",
            "React",
            "Angular",
            "Node.js",
            "Java",
            "C++",
            "SQL",
            "MongoDB",
            "Cybersecurity"
        ]
        
        selected_tech = st.selectbox(
            "🔧 Select Technology",
            tech_options,
            index=0
        )
        
        # Custom technology input
        use_custom = st.checkbox("Use Custom Technology")
        if use_custom:
            custom_tech = st.text_input("Enter Custom Technology", placeholder="e.g., Blockchain, Rust, Go")
            if custom_tech:
                selected_tech = custom_tech
        
        st.markdown("---")
        
        # Number of questions (up to 1000)
        num_questions = st.select_slider(
            "📊 Number of Questions",
            options=[5, 10, 20, 30, 50, 100, 200, 300, 500, 1000],
            value=20
        )
        
        st.markdown("---")
        
        # Company selection with 10+ options
        st.markdown("### 🏢 Company-Specific Prep")
        company_options = [
            "Select Company",
            "Google", 
            "Amazon", 
            "Microsoft", 
            "Meta (Facebook)", 
            "Apple",
            "Netflix",
            "Tesla",
            "Uber",
            "Airbnb",
            "LinkedIn",
            "Infosys", 
            "TCS", 
            "Wipro", 
            "Accenture", 
            "Cognizant",
            "IBM",
            "Oracle",
            "Salesforce"
        ]
        
        selected_company = st.selectbox("Select Company", company_options)
        
        # Custom company
        use_custom_company = st.checkbox("Use Custom Company")
        if use_custom_company:
            custom_company = st.text_input("Enter Company Name", placeholder="e.g., Deloitte, Stripe")
            if custom_company:
                selected_company = custom_company
        
        st.markdown("---")
        
        # Info section
        st.markdown("### 💡 Tips")
        st.info("🌐 Real-time web scraping\n\n📚 Multiple sources\n\n🎥 Video resources\n\n💾 Export to PDF/Word")
    
    # Main content area
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>🎯</h3><p>Technology</p><h4>' + selected_tech + '</h4></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><h3>📚</h3><p>Questions</p><h4>' + str(num_questions) + '</h4></div>', unsafe_allow_html=True)
    
    with col3:
        company_display = selected_company if selected_company != "Select Company" else "General"
        st.markdown('<div class="metric-card"><h3>🏢</h3><p>Company</p><h4>' + company_display + '</h4></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Scrape & Generate Questions", use_container_width=True):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🌐 Initializing web scraper...")
            progress_bar.progress(10)
            time.sleep(0.5)
            
            status_text.text(f"🔍 Searching for {selected_tech} interview questions...")
            progress_bar.progress(30)
            
            # Scrape questions
            company_for_scrape = selected_company if selected_company != "Select Company" else None
            questions_data = scrape_interview_questions(selected_tech, num_questions, company_for_scrape)
            
            progress_bar.progress(70)
            status_text.text("📊 Processing and formatting data...")
            time.sleep(0.5)
            
            progress_bar.progress(90)
            status_text.text("✅ Finalizing questions...")
            time.sleep(0.5)
            
            # Store in session state
            st.session_state['questions_data'] = questions_data
            st.session_state['tech'] = selected_tech
            st.session_state['company'] = selected_company
            
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            
            st.success(f"✅ Successfully scraped {len(questions_data)} questions from the web!")
            st.balloons()
    
    # Display questions
    if 'questions_data' in st.session_state and st.session_state['questions_data']:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">📋 Interview Questions & Answers</h2>', unsafe_allow_html=True)
        
        questions_data = st.session_state['questions_data']
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Questions", len(questions_data))
        with col2:
            sources = set([q.get('source', 'Unknown') for q in questions_data])
            st.metric("Sources", len(sources))
        with col3:
            with_answers = sum(1 for q in questions_data if q.get('answer'))
            st.metric("With Answers", with_answers)
        with col4:
            with_videos = sum(1 for q in questions_data if q.get('video'))
            st.metric("Video Links", with_videos)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display each question
        for idx, qa in enumerate(questions_data, 1):
            with st.container():
                # Question card
                st.markdown(f"""
                <div class="question-card">
                    <div class="question-text">Q{idx}. {qa['question']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Answer card
                if qa.get('answer'):
                    st.markdown(f"""
                    <div class="answer-card">
                        <div class="answer-text">
                            <strong>Answer:</strong><br>
                            {qa['answer'][:800]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Source and URL badges
                col1, col2 = st.columns(2)
                with col1:
                    if qa.get('source'):
                        st.markdown(f"""
                        <span class="source-badge">📚 {qa['source']}</span>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if qa.get('url'):
                        st.markdown(f"""
                        <a href="{qa['url']}" target="_blank" class="source-badge">🔗 View Source</a>
                        """, unsafe_allow_html=True)
                
                # Video link
                if qa.get('video'):
                    st.markdown(f"""
                    <a href="{qa['video']}" target="_blank" class="video-link">
                        🎥 Watch Video Explanation
                    </a>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
        
        # Export options
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">💾 Export Options</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("<br>", unsafe_allow_html=True)
        
        with col2:
            # PDF download
            with st.spinner("Generating PDF..."):
                pdf_buffer = create_pdf(
                    questions_data, 
                    st.session_state['tech'],
                    st.session_state['company'] if st.session_state['company'] != "Select Company" else None
                )
            
            filename_base = f"{st.session_state['tech'].replace(' ', '_')}_Interview_Questions"
            if st.session_state['company'] != "Select Company":
                filename_base += f"_{st.session_state['company'].replace(' ', '_')}"
            
            st.download_button(
                label="📄 Download as PDF",
                data=pdf_buffer,
                file_name=f"{filename_base}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        with col3:
            # Word download
            with st.spinner("Generating Word document..."):
                word_buffer = create_word(
                    questions_data,
                    st.session_state['tech'],
                    st.session_state['company'] if st.session_state['company'] != "Select Company" else None
                )
            
            st.download_button(
                label="📝 Download as Word",
                data=word_buffer,
                file_name=f"{filename_base}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
    
    elif 'questions_data' in st.session_state and not st.session_state['questions_data']:
        st.warning("⚠️ No questions were found. Try a different technology or check your internet connection.")
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🌟 <strong>Interview Prep Master - Dynamic Web Scraper</strong></p>
        <p>🌐 Real-time scraping from GeeksforGeeks, InterviewBit, JavaTpoint, Guru99 & more</p>
        <p>💼 Practice • 🎯 Prepare • 🚀 Succeed</p>
        <p style="font-size: 0.9rem; margin-top: 10px;">
            <strong>Sources:</strong> GeeksforGeeks | InterviewBit | JavaTpoint | Guru99 | Web Search<br>
            <strong>Note:</strong> Questions are dynamically scraped from public websites. Please verify answers independently.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

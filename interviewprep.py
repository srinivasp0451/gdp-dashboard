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
from xml.sax.saxutils import escape
from urllib.parse import quote_plus, urljoin
import html

# Page configuration
st.set_page_config(
    page_title="Interview Prep Master Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with scroll to top button
st.markdown("""
<style>
    .main .block-container {
        max-height: 90vh;
        overflow-y: auto;
        padding-bottom: 50px;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 20px 0;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #667eea;
        font-weight: 600;
        margin-top: 3rem;
        margin-bottom: 2rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    
    .question-container {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 25px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    
    .question-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .question-number {
        font-size: 1rem;
        font-weight: 600;
        opacity: 0.9;
        margin-bottom: 10px;
    }
    
    .question-text {
        font-size: 1.3rem;
        font-weight: 600;
        line-height: 1.6;
        margin: 0;
    }
    
    .answer-section {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 4px solid #48bb78;
    }
    
    .answer-label {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 15px;
        display: block;
    }
    
    .answer-text {
        font-size: 1.05rem;
        color: #4a5568;
        line-height: 1.8;
        text-align: justify;
    }
    
    .answer-text ul {
        margin: 15px 0;
        padding-left: 25px;
    }
    
    .answer-text li {
        margin: 10px 0;
        line-height: 1.6;
    }
    
    .metadata-section {
        display: flex;
        gap: 15px;
        flex-wrap: wrap;
        margin: 20px 0;
        padding: 15px;
        background: #e6f7ff;
        border-radius: 8px;
    }
    
    .badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 5px;
    }
    
    .badge-source {
        background: #48bb78;
        color: white;
    }
    
    .badge-difficulty {
        background: #f6ad55;
        color: white;
    }
    
    .badge-category {
        background: #667eea;
        color: white;
    }
    
    .link-button {
        display: inline-block;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        margin: 10px 5px;
        transition: transform 0.2s;
    }
    
    .link-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-top: 4px solid #667eea;
    }
    
    .stButton>button {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 40px 0;
    }
    
    /* Scroll to top button */
    #scrollTopBtn {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 99;
        border: none;
        outline: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        cursor: pointer;
        padding: 15px 20px;
        border-radius: 50%;
        font-size: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: all 0.3s;
    }
    
    #scrollTopBtn:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #667eea;
    }
    
    .scraping-progress {
        background: #e6f7ff;
        border-left: 4px solid #1890ff;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>

<!-- Scroll to Top Button -->
<button onclick="scrollToTop()" id="scrollTopBtn" title="Go to top">↑</button>

<script>
// Show button when scrolling down
window.onscroll = function() {
    var btn = document.getElementById("scrollTopBtn");
    if (document.body.scrollTop > 200 || document.documentElement.scrollTop > 200) {
        btn.style.display = "block";
    } else {
        btn.style.display = "none";
    }
};

// Smooth scroll to top
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}
</script>
""", unsafe_allow_html=True)


# Real web scraping functions
def scrape_with_requests(url, timeout=10):
    """Helper function to make HTTP requests with proper headers"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout, verify=False)
        if response.status_code == 200:
            return BeautifulSoup(response.content, 'html.parser')
    except:
        pass
    return None


def scrape_interviewbit(technology):
    """Scrape InterviewBit"""
    questions = []
    tech_map = {
        "Artificial Intelligence": "artificial-intelligence",
        "Machine Learning": "machine-learning",
        "Python": "python",
        "JavaScript": "javascript",
        "Java": "java",
        "React": "react",
        "Node.js": "node-js",
        "SQL": "sql"
    }
    
    tech_slug = tech_map.get(technology, technology.lower().replace(" ", "-"))
    url = f"https://www.interviewbit.com/{tech_slug}-interview-questions/"
    
    st.info(f"🔍 Searching InterviewBit for {technology} questions...")
    
    soup = scrape_with_requests(url)
    if soup:
        # Find all question elements
        question_elements = soup.find_all(['h2', 'h3', 'div'], class_=re.compile('question|title|heading'))
        
        for elem in question_elements[:20]:
            text = elem.get_text(strip=True)
            if len(text) > 25 and ('?' in text or any(word in text.lower() for word in ['what', 'explain', 'how', 'describe'])):
                # Try to find answer
                answer_elem = elem.find_next(['p', 'div'])
                answer = answer_elem.get_text(strip=True) if answer_elem else ""
                
                if len(answer) < 50:
                    answer = f"This is a comprehensive {technology} interview question. Key points to cover include understanding the fundamental concepts, practical applications, and real-world implementations."
                
                questions.append({
                    'question': text,
                    'answer': answer[:1500],
                    'source': 'InterviewBit',
                    'url': url,
                    'video': f"https://www.youtube.com/results?search_query={quote_plus(technology + ' ' + text[:50])}"
                })
        
        if questions:
            st.success(f"✅ Found {len(questions)} questions from InterviewBit")
    
    return questions


def scrape_geeksforgeeks(technology):
    """Scrape GeeksforGeeks"""
    questions = []
    tech_map = {
        "Artificial Intelligence": "ai",
        "Machine Learning": "machine-learning",
        "Python": "python",
        "JavaScript": "javascript",
        "Java": "java",
        "React": "reactjs",
        "Node.js": "node-js",
        "SQL": "sql"
    }
    
    tech_slug = tech_map.get(technology, technology.lower().replace(" ", "-"))
    url = f"https://www.geeksforgeeks.org/{tech_slug}-interview-questions/"
    
    st.info(f"🔍 Searching GeeksforGeeks for {technology} questions...")
    
    soup = scrape_with_requests(url)
    if soup:
        # Find question-answer pairs
        articles = soup.find_all(['article', 'div'], class_=re.compile('content|article'))
        
        for article in articles[:15]:
            headings = article.find_all(['h2', 'h3'])
            for heading in headings:
                text = heading.get_text(strip=True)
                if len(text) > 20 and ('?' in text or text.lower().startswith(('what', 'how', 'explain', 'why'))):
                    # Find answer in next paragraph
                    answer_elem = heading.find_next(['p', 'div'])
                    answer = ""
                    if answer_elem:
                        answer = answer_elem.get_text(strip=True)
                    
                    if len(answer) < 50:
                        answer = f"Understanding {technology} concepts is crucial. This question tests your knowledge of core principles and practical implementation."
                    
                    questions.append({
                        'question': text,
                        'answer': answer[:1500],
                        'source': 'GeeksforGeeks',
                        'url': url,
                        'video': f"https://www.youtube.com/results?search_query={quote_plus(text)}"
                    })
        
        if questions:
            st.success(f"✅ Found {len(questions)} questions from GeeksforGeeks")
    
    return questions


def scrape_medium(technology):
    """Scrape Medium articles"""
    questions = []
    search_url = f"https://medium.com/search?q={quote_plus(technology + ' interview questions')}"
    
    st.info(f"🔍 Searching Medium for {technology} articles...")
    
    soup = scrape_with_requests(search_url)
    if soup:
        # Find article titles
        articles = soup.find_all(['h2', 'h3'], limit=15)
        
        for article in articles:
            text = article.get_text(strip=True)
            if len(text) > 30 and technology.lower() in text.lower():
                # Try to get link
                link_elem = article.find_parent('a')
                article_url = link_elem.get('href', search_url) if link_elem else search_url
                
                questions.append({
                    'question': text,
                    'answer': f"This Medium article discusses important {technology} interview concepts. Read the full article for comprehensive coverage of this topic with real-world examples and best practices.",
                    'source': 'Medium',
                    'url': article_url if article_url.startswith('http') else f"https://medium.com{article_url}",
                    'video': f"https://www.youtube.com/results?search_query={quote_plus(text[:60])}"
                })
        
        if questions:
            st.success(f"✅ Found {len(questions)} articles from Medium")
    
    return questions


def scrape_stackoverflow(technology):
    """Scrape Stack Overflow questions"""
    questions = []
    search_url = f"https://stackoverflow.com/search?q={quote_plus(technology)}"
    
    st.info(f"🔍 Searching Stack Overflow for {technology} questions...")
    
    soup = scrape_with_requests(search_url)
    if soup:
        # Find questions
        question_summaries = soup.find_all('div', class_='question-summary', limit=20)
        
        for summary in question_summaries:
            title_elem = summary.find('a', class_='question-hyperlink')
            if title_elem:
                title = title_elem.get_text(strip=True)
                q_url = 'https://stackoverflow.com' + title_elem.get('href', '')
                
                # Get excerpt
                excerpt_elem = summary.find('div', class_='excerpt')
                excerpt = excerpt_elem.get_text(strip=True) if excerpt_elem else ""
                
                if len(title) > 20:
                    questions.append({
                        'question': title,
                        'answer': excerpt if excerpt else f"This Stack Overflow question has community-verified answers. Visit the link to see detailed explanations and code examples.",
                        'source': 'Stack Overflow',
                        'url': q_url,
                        'video': f"https://www.youtube.com/results?search_query={quote_plus(technology + ' ' + title[:50])}"
                    })
        
        if questions:
            st.success(f"✅ Found {len(questions)} questions from Stack Overflow")
    
    return questions


def scrape_github_repos(technology):
    """Scrape GitHub awesome lists"""
    questions = []
    search_url = f"https://github.com/search?q={quote_plus(technology + ' interview questions')}&type=repositories"
    
    st.info(f"🔍 Searching GitHub for {technology} resources...")
    
    soup = scrape_with_requests(search_url)
    if soup:
        # Find repository links
        repos = soup.find_all('a', class_='v-align-middle', limit=10)
        
        for repo in repos:
            text = repo.get_text(strip=True)
            if len(text) > 10:
                repo_url = 'https://github.com' + repo.get('href', '')
                
                questions.append({
                    'question': f"Explore: {text}",
                    'answer': f"This GitHub repository contains curated {technology} interview questions and answers. It's a comprehensive resource maintained by the community with regular updates and contributions.",
                    'source': 'GitHub',
                    'url': repo_url,
                    'video': f"https://www.youtube.com/results?search_query={quote_plus(technology + ' tutorial')}"
                })
        
        if questions:
            st.success(f"✅ Found {len(questions)} repositories from GitHub")
    
    return questions


def generate_fallback_questions(technology, count=50):
    """Generate intelligent fallback questions when scraping fails"""
    
    question_templates = [
        f"What is {technology} and what are its key features?",
        f"Explain the core concepts of {technology}.",
        f"What are the main advantages of using {technology}?",
        f"What are the common use cases for {technology}?",
        f"How does {technology} compare to similar technologies?",
        f"What are the best practices when working with {technology}?",
        f"Describe the architecture of {technology}.",
        f"What are the latest trends and updates in {technology}?",
        f"How do you optimize performance in {technology}?",
        f"What are the common pitfalls to avoid in {technology}?",
        f"Explain the ecosystem and tools around {technology}.",
        f"What are the security considerations for {technology}?",
        f"How do you debug and troubleshoot issues in {technology}?",
        f"What are the scalability challenges with {technology}?",
        f"Describe the learning curve and resources for {technology}.",
        f"What are real-world applications of {technology}?",
        f"How do you handle errors and exceptions in {technology}?",
        f"What testing strategies work best for {technology}?",
        f"Explain the deployment process for {technology} applications.",
        f"What are the performance optimization techniques in {technology}?",
        f"How do you manage state/data in {technology}?",
        f"What are the design patterns commonly used in {technology}?",
        f"Explain the integration capabilities of {technology}.",
        f"What are the monitoring and logging best practices for {technology}?",
        f"How do you ensure code quality in {technology} projects?",
        f"What are the career opportunities and certifications in {technology}?",
        f"Describe the community and ecosystem support for {technology}.",
        f"What are the limitations and constraints of {technology}?",
        f"How do you stay updated with {technology} developments?",
        f"What projects showcase advanced {technology} usage?"
    ]
    
    questions = []
    for i, template in enumerate(question_templates[:count]):
        difficulty = ["Easy", "Medium", "Hard"][i % 3]
        category = ["Fundamentals", "Advanced", "Architecture", "Best Practices"][i % 4]
        
        questions.append({
            'question': template,
            'answer': f"This is an important {technology} interview question. The answer should cover:\n\n• Core concepts and definitions\n• Practical applications and examples\n• Best practices and recommendations\n• Common patterns and anti-patterns\n• Real-world use cases\n\nResearch this topic thoroughly and practice explaining it clearly and concisely.",
            'source': f'{technology} Knowledge Base',
            'difficulty': difficulty,
            'category': category,
            'url': f"https://www.google.com/search?q={quote_plus(template)}",
            'video': f"https://www.youtube.com/results?search_query={quote_plus(template)}"
        })
    
    return questions


def scrape_interview_questions(technology, num_questions, filter_type="all", company=None):
    """Main scraping orchestrator with multiple sources"""
    
    all_questions = []
    
    # Try multiple sources
    scrapers = [
        ("InterviewBit", lambda: scrape_interviewbit(technology)),
        ("GeeksforGeeks", lambda: scrape_geeksforgeeks(technology)),
        ("Medium", lambda: scrape_medium(technology)),
        ("Stack Overflow", lambda: scrape_stackoverflow(technology)),
        ("GitHub", lambda: scrape_github_repos(technology)),
    ]
    
    for source_name, scraper_func in scrapers:
        try:
            questions = scraper_func()
            all_questions.extend(questions)
            time.sleep(1)  # Be respectful to servers
        except Exception as e:
            st.warning(f"⚠️ Could not scrape {source_name}: {str(e)}")
            continue
    
    # Remove duplicates
    unique_questions = []
    seen = set()
    
    for q in all_questions:
        q_lower = q['question'].lower()[:100]
        if q_lower not in seen and len(q['question']) > 20:
            seen.add(q_lower)
            unique_questions.append(q)
    
    # If not enough questions, add fallback
    if len(unique_questions) < num_questions:
        st.info(f"📚 Adding {num_questions - len(unique_questions)} knowledge-based questions...")
        fallback = generate_fallback_questions(technology, num_questions - len(unique_questions))
        unique_questions.extend(fallback)
    
    # Add company tags
    if company and company != "Select Company":
        for q in unique_questions:
            q['company'] = company
            q['question'] = f"[{company}] {q['question']}"
    
    return unique_questions[:num_questions]


def create_pdf(questions_data, technology, company=None):
    """Create PDF"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    content = []
    
    title = Paragraph(escape(f"Interview Questions: {technology}"), styles['Title'])
    content.append(title)
    content.append(Spacer(1, 0.3*inch))
    
    for idx, qa in enumerate(questions_data, 1):
        q_text = f"<b>Q{idx}.</b> {escape(qa['question'][:400])}"
        content.append(Paragraph(q_text, styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))
        
        if qa.get('answer'):
            answer_clean = html.unescape(qa['answer']).replace('\n', ' ')[:800]
            content.append(Paragraph(escape(answer_clean), styles['BodyText']))
        
        if qa.get('source'):
            content.append(Paragraph(f"<i>Source: {escape(qa['source'])}</i>", styles['Normal']))
        
        content.append(Spacer(1, 0.2*inch))
        
        if idx % 3 == 0:
            content.append(PageBreak())
    
    doc.build(content)
    buffer.seek(0)
    return buffer


def create_word(questions_data, technology, company=None):
    """Create Word document"""
    doc = Document()
    
    title = doc.add_heading(f'Interview Questions: {technology}', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()
    
    for idx, qa in enumerate(questions_data, 1):
        doc.add_heading(f"Question {idx}", level=2)
        doc.add_paragraph(qa['question'])
        
        if qa.get('answer'):
            doc.add_paragraph("Answer:", style='Heading 3')
            answer_clean = html.unescape(qa['answer'])[:1000]
            doc.add_paragraph(answer_clean)
        
        if qa.get('source'):
            p = doc.add_paragraph(f"Source: {qa['source']}")
            p.runs[0].font.color.rgb = RGBColor(72, 187, 120)
        
        doc.add_paragraph()
    
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


# Main App
def main():
    st.markdown('<h1 class="main-header">🎯 Interview Prep Master Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 30px;">Dynamic Web Scraping • Real Questions • Beautiful UI</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        
        tech_options = [
            "Artificial Intelligence", "Machine Learning", "Python", "JavaScript",
            "Java", "C++", "C#", "React", "Angular", "Vue.js", "Node.js",
            "Django", "Flask", "Spring Boot", "Azure Cloud", "AWS Cloud",
            "GCP Cloud", "Docker", "Kubernetes", "DevOps", "Data Science",
            "Data Engineering", "SQL", "MongoDB", "PostgreSQL", "Redis",
            "Kafka", "Spark", "Hadoop", "Cybersecurity", "Blockchain"
        ]
        
        selected_tech = st.selectbox("🔧 Technology", tech_options, index=0)
        
        use_custom = st.checkbox("✏️ Custom Technology")
        if use_custom:
            custom_tech = st.text_input("Enter Technology", placeholder="e.g., Rust, Go")
            if custom_tech:
                selected_tech = custom_tech
        
        st.markdown("---")
        
        num_questions = st.select_slider(
            "📊 Questions",
            options=[10, 20, 30, 50, 100, 200, 300, 500, 1000],
            value=50
        )
        
        st.markdown("---")
        
        filter_type = st.radio(
            "🔍 Filter",
            ["All Questions", "Trending", "Latest"],
            index=0
        )
        
        st.markdown("---")
        
        company_options = [
            "Select Company", "Google", "Amazon", "Microsoft", "Meta",
            "Apple", "Netflix", "Tesla", "Uber", "Airbnb", "LinkedIn",
            "Salesforce", "Oracle", "IBM", "Adobe", "Infosys", "TCS",
            "Wipro", "Accenture", "Cognizant"
        ]
        
        selected_company = st.selectbox("🏢 Company", company_options)
        
        use_custom_company = st.checkbox("✏️ Custom Company")
        if use_custom_company:
            custom_company = st.text_input("Company Name")
            if custom_company:
                selected_company = custom_company
        
        st.markdown("---")
        st.success("✨ Features:\n\n• Real-time scraping\n• Multiple sources\n• Beautiful formatting\n• Scroll to top ↑\n• PDF/Word export")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><p>🎯 Technology</p><h3>{selected_tech}</h3></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><p>📚 Questions</p><h3>{num_questions}</h3></div>', unsafe_allow_html=True)
    with col3:
        company_display = selected_company if selected_company != "Select Company" else "General"
        st.markdown(f'<div class="metric-card"><p>🏢 Company</p><h3>{company_display}</h3></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><p>🔍 Filter</p><h3>{filter_type}</h3></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Generate Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Scrape & Generate Questions", use_container_width=True):
            progress_bar = st.progress(0)
            status = st.empty()
            
            status.text("🌐 Initializing web scrapers...")
            progress_bar.progress(20)
            
            company_for_scrape = selected_company if selected_company != "Select Company" else None
            questions_data = scrape_interview_questions(
                selected_tech,
                num_questions,
                filter_type.lower(),
                company_for_scrape
            )
            
            progress_bar.progress(80)
            status.text("✅ Processing results...")
            time.sleep(0.5)
            
            st.session_state['questions_data'] = questions_data
            st.session_state['tech'] = selected_tech
            st.session_state['company'] = selected_company
            
            progress_bar.progress(100)
            status.empty()
            progress_bar.empty()
            
            st.success(f"✅ Successfully generated {len(questions_data)} questions!")
            st.balloons()
    
    # Display Questions
    if 'questions_data' in st.session_state and st.session_state['questions_data']:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">📋 Interview Questions & Answers</h2>', unsafe_allow_html=True)
        
        questions_data = st.session_state['questions_data']
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📚 Total", len(questions_data))
        with col2:
            sources = len(set([q.get('source', 'Unknown') for q in questions_data]))
            st.metric("🌐 Sources", sources)
        with col3:
            with_answers = sum(1 for q in questions_data if q.get('answer'))
            st.metric("✅ Answered", with_answers)
        with col4:
            with_videos = sum(1 for q in questions_data if q.get('video'))
            st.metric("🎥 Videos", with_videos)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display each question with beautiful formatting
        for idx, qa in enumerate(questions_data, 1):
            st.markdown(f"""
            <div class="question-container">
                <div class="question-header">
                    <div class="question-number">Question {idx} of {len(questions_data)}</div>
                    <div class="question-text">{qa['question']}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Answer section with proper formatting
            if qa.get('answer'):
                answer = qa['answer']
                
                # Format bullet points and paragraphs
                formatted_answer = ""
                paragraphs = answer.split('\n\n')
                
                for para in paragraphs:
                    if para.strip():
                        # Check if it's a bullet list
                        if '•' in para or para.strip().startswith('-'):
                            items = [item.strip('• -') for item in para.split('\n') if item.strip()]
                            formatted_answer += "<ul style='margin: 15px 0; padding-left: 25px;'>"
                            for item in items:
                                if item:
                                    item = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', item)
                                    formatted_answer += f"<li style='margin: 10px 0; line-height: 1.6;'>{item}</li>"
                            formatted_answer += "</ul>"
                        else:
                            # Regular paragraph
                            para = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', para)
                            formatted_answer += f"<p style='margin: 15px 0; line-height: 1.8;'>{para}</p>"
                
                st.markdown(f"""
                <div class="answer-section">
                    <span class="answer-label">📝 Answer:</span>
                    <div class="answer-text">{formatted_answer}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Metadata section
            st.markdown('<div class="metadata-section">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if qa.get('source'):
                    st.markdown(f'<span class="badge badge-source">📚 {qa["source"]}</span>', unsafe_allow_html=True)
            
            with col2:
                if qa.get('difficulty'):
                    st.markdown(f'<span class="badge badge-difficulty">⚡ {qa["difficulty"]}</span>', unsafe_allow_html=True)
            
            with col3:
                if qa.get('category'):
                    st.markdown(f'<span class="badge badge-category">🏷️ {qa["category"]}</span>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Links
            link_col1, link_col2 = st.columns(2)
            
            with link_col1:
                if qa.get('url'):
                    st.markdown(f'<a href="{qa["url"]}" target="_blank" class="link-button">🔗 View Source</a>', unsafe_allow_html=True)
            
            with link_col2:
                if qa.get('video'):
                    st.markdown(f'<a href="{qa["video"]}" target="_blank" class="link-button">🎥 Watch Video</a>', unsafe_allow_html=True)
            
            st.markdown('</div><br>', unsafe_allow_html=True)
        
        # Export Section
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">💾 Export Your Questions</h2>', unsafe_allow_html=True)
        
        st.info("📥 Download your personalized question set in PDF or Word format for offline practice.")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            with st.spinner("Generating PDF..."):
                pdf_buffer = create_pdf(
                    questions_data,
                    st.session_state['tech'],
                    st.session_state.get('company')
                )
            
            filename = f"{st.session_state['tech'].replace(' ', '_')}_Questions"
            if st.session_state.get('company') and st.session_state['company'] != "Select Company":
                filename += f"_{st.session_state['company'].replace(' ', '_')}"
            
            st.download_button(
                "📄 Download PDF",
                pdf_buffer,
                f"{filename}.pdf",
                "application/pdf",
                use_container_width=True
            )
        
        with col3:
            with st.spinner("Generating Word..."):
                word_buffer = create_word(
                    questions_data,
                    st.session_state['tech'],
                    st.session_state.get('company')
                )
            
            st.download_button(
                "📝 Download Word",
                word_buffer,
                f"{filename}.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
    
    # Footer
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 30px;">
        <p style="font-size: 1.3rem; font-weight: 600; margin-bottom: 15px;">🌟 Interview Prep Master Pro</p>
        <p style="font-size: 1rem;">Real-time Web Scraping • Multiple Sources • Beautiful Formatting</p>
        <p style="font-size: 0.95rem; margin: 15px 0;">
            🌐 Sources: InterviewBit • GeeksforGeeks • Medium • Stack Overflow • GitHub
        </p>
        <p style="font-size: 0.9rem; color: #999;">
            💼 Practice Daily • 🎯 Stay Focused • 🚀 Achieve Success
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

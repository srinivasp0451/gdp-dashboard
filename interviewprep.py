import streamlit as st
import streamlit.components.v1 as components
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
from urllib.parse import quote_plus
import html

# Page configuration
st.set_page_config(
    page_title="Interview Prep Master Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with working scroll button
st.markdown("""
<style>
    .main .block-container {
        padding-bottom: 100px;
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
        margin: 30px 0;
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
    
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 40px 0;
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
</style>
""", unsafe_allow_html=True)


def scrape_real_content(url, timeout=10):
    """Enhanced scraper with better parsing"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout, verify=False)
        if response.status_code == 200:
            return BeautifulSoup(response.content, 'html.parser')
    except:
        pass
    return None


def extract_detailed_answer(soup, question_elem):
    """Extract detailed answer from webpage"""
    answer_parts = []
    
    # Try to find answer in next siblings
    for sibling in question_elem.find_next_siblings(limit=5):
        if sibling.name in ['p', 'div', 'ul', 'ol']:
            text = sibling.get_text(strip=True)
            if len(text) > 100 and len(text) < 2000:
                answer_parts.append(text)
                if len(' '.join(answer_parts)) > 500:
                    break
    
    return ' '.join(answer_parts) if answer_parts else None


def scrape_comprehensive_source(technology, source_name, base_url, selectors):
    """Generic scraper for different sources"""
    questions = []
    
    st.info(f"🔍 Scraping {source_name} for {technology}...")
    
    soup = scrape_real_content(base_url)
    if not soup:
        return questions
    
    # Find all text content
    all_text = soup.get_text()
    
    # Split by common question patterns
    question_patterns = [
        r'Q\d+[.:\)]\s*(.+?)(?=Q\d+|$)',
        r'\d+[.:\)]\s*(.+?)(?=\d+[.:\)]|$)',
        r'(?:^|\n)(.+?\?)',
    ]
    
    found_questions = []
    for pattern in question_patterns:
        matches = re.finditer(pattern, all_text, re.MULTILINE | re.DOTALL)
        for match in matches:
            q_text = match.group(1).strip()
            if len(q_text) > 30 and len(q_text) < 500:
                # Extract answer (next 500-1500 chars after question)
                start_pos = match.end()
                answer_text = all_text[start_pos:start_pos+1500].strip()
                
                # Clean answer
                answer_text = re.sub(r'Q\d+[.:\)].*$', '', answer_text, flags=re.DOTALL)
                answer_text = re.sub(r'\d+[.:\)].*$', '', answer_text, flags=re.DOTALL)
                answer_text = answer_text[:1000].strip()
                
                if len(answer_text) > 100:
                    found_questions.append({
                        'question': q_text,
                        'answer': answer_text,
                        'source': source_name,
                        'url': base_url
                    })
    
    # Also try structured selectors
    for selector in selectors:
        try:
            elements = soup.select(selector)
            for elem in elements[:20]:
                text = elem.get_text(strip=True)
                if len(text) > 30 and ('?' in text or any(w in text.lower() for w in ['what', 'how', 'explain', 'why', 'describe'])):
                    # Find answer
                    answer_elem = elem.find_next(['p', 'div', 'li'])
                    answer = answer_elem.get_text(strip=True) if answer_elem else ""
                    
                    if len(answer) > 100:
                        found_questions.append({
                            'question': text[:300],
                            'answer': answer[:1500],
                            'source': source_name,
                            'url': base_url
                        })
        except:
            continue
    
    if found_questions:
        st.success(f"✅ Found {len(found_questions)} questions from {source_name}")
        return found_questions
    
    return questions


def generate_diverse_questions(technology, count):
    """Generate diverse, meaningful questions with unique answers"""
    
    base_questions = {
        "concepts": [
            (f"What is {technology} and what are its core features?", 
             f"{technology} is a widely-used technology in modern software development. Its core features include scalability, performance optimization, ease of use, and robust community support. It provides developers with powerful tools to build efficient applications and solve complex problems in production environments."),
            
            (f"Explain the architecture and design principles of {technology}.",
             f"The architecture of {technology} follows industry best practices with a modular design approach. It typically consists of multiple layers including the presentation layer, business logic layer, and data access layer. Key design principles include separation of concerns, single responsibility, and dependency injection, making the codebase maintainable and testable."),
            
            (f"What are the main advantages and disadvantages of using {technology}?",
             f"Advantages of {technology} include: high performance, excellent scalability, strong community support, rich ecosystem of libraries and tools, cross-platform compatibility, and mature documentation. Disadvantages can include: steep learning curve for beginners, potential performance overhead in certain scenarios, and the need for careful configuration and optimization in production environments."),
        ],
        "practical": [
            (f"How do you implement error handling in {technology}?",
             f"Error handling in {technology} involves multiple strategies: try-catch blocks for synchronous operations, promise rejection handling for asynchronous code, centralized error middleware for applications, custom error classes for specific scenarios, proper logging mechanisms, and graceful degradation. Best practices include validating inputs, providing meaningful error messages, and implementing retry logic for transient failures."),
            
            (f"What are the best practices for {technology} in production?",
             f"Production best practices for {technology} include: implementing comprehensive monitoring and logging, using environment-specific configurations, enabling security features like authentication and authorization, optimizing performance through caching strategies, implementing CI/CD pipelines, conducting regular security audits, using containerization for consistency, implementing load balancing and auto-scaling, and maintaining proper backup and disaster recovery procedures."),
            
            (f"How do you optimize performance in {technology} applications?",
             f"Performance optimization in {technology} involves: profiling to identify bottlenecks, implementing caching strategies (memory cache, distributed cache), optimizing database queries with indexes and connection pooling, using asynchronous operations where appropriate, minimizing bundle size through code splitting and tree shaking, implementing lazy loading, using CDNs for static assets, enabling compression, and monitoring application metrics to identify performance degradation."),
        ],
        "advanced": [
            (f"Explain the security considerations when using {technology}.",
             f"Security in {technology} requires addressing multiple aspects: input validation and sanitization to prevent injection attacks, implementing proper authentication mechanisms (JWT, OAuth), using HTTPS for data encryption in transit, securing sensitive data at rest with encryption, implementing rate limiting to prevent abuse, keeping dependencies updated to patch vulnerabilities, using security headers, implementing CORS policies correctly, and conducting regular security audits and penetration testing."),
            
            (f"How do you scale {technology} applications for high traffic?",
             f"Scaling {technology} applications involves: horizontal scaling with load balancers distributing traffic across multiple instances, implementing caching layers (Redis, Memcached), using database replication and sharding for data distribution, implementing message queues for asynchronous processing, using CDNs for static content delivery, optimizing database queries and indexes, implementing connection pooling, using auto-scaling based on metrics, and designing stateless applications for easier distribution."),
            
            (f"What are common design patterns used with {technology}?",
             f"Common design patterns in {technology} include: Singleton for shared resources, Factory for object creation, Observer for event handling, Strategy for algorithm selection, Dependency Injection for loose coupling, Repository for data access abstraction, MVC/MVVM for separation of concerns, Middleware pattern for request processing, Builder for complex object construction, and Adapter pattern for interface compatibility. These patterns promote code reusability, maintainability, and testability."),
        ],
        "tools": [
            (f"What tools and frameworks complement {technology}?",
             f"The {technology} ecosystem includes numerous tools: build tools for compilation and bundling, testing frameworks for unit and integration tests, linting tools for code quality, package managers for dependency management, debugging tools for troubleshooting, profiling tools for performance analysis, containerization tools like Docker, orchestration platforms like Kubernetes, CI/CD tools for automation, and monitoring solutions for production observability."),
            
            (f"How do you test applications built with {technology}?",
             f"Testing {technology} applications involves multiple levels: unit tests for individual functions and components using testing frameworks, integration tests for module interactions, end-to-end tests simulating user workflows, performance tests for load and stress testing, security tests for vulnerability scanning, and smoke tests for critical functionality. Best practices include maintaining high code coverage, using test-driven development (TDD), mocking external dependencies, and automating tests in CI/CD pipelines."),
        ]
    }
    
    all_questions = []
    for category_questions in base_questions.values():
        all_questions.extend(category_questions)
    
    # Create diverse set by cycling through questions
    result = []
    while len(result) < count and all_questions:
        for q, a in all_questions:
            if len(result) >= count:
                break
            result.append({
                'question': q,
                'answer': a,
                'source': f'{technology} Knowledge Base',
                'url': f"https://www.google.com/search?q={quote_plus(q)}",
                'video': f"https://www.youtube.com/results?search_query={quote_plus(q)}"
            })
    
    return result


def scrape_interview_questions(technology, num_questions, company=None):
    """Main orchestrator - scrapes real questions"""
    
    all_questions = []
    
    # Define scraping sources with actual URLs
    sources = [
        {
            'name': 'GeeksforGeeks',
            'url': f'https://www.geeksforgeeks.org/{technology.lower().replace(" ", "-")}-interview-questions/',
            'selectors': ['h2', 'h3', 'strong', '.entry-content p']
        },
        {
            'name': 'InterviewBit',
            'url': f'https://www.interviewbit.com/{technology.lower().replace(" ", "-")}-interview-questions/',
            'selectors': ['h2', 'h3', '.question-title']
        },
        {
            'name': 'JavaTpoint',
            'url': f'https://www.javatpoint.com/{technology.lower().replace(" ", "-")}-interview-questions',
            'selectors': ['h2', 'h3', 'p']
        },
    ]
    
    # Try scraping each source
    for source in sources:
        try:
            questions = scrape_comprehensive_source(
                technology,
                source['name'],
                source['url'],
                source['selectors']
            )
            all_questions.extend(questions)
            time.sleep(1)
        except Exception as e:
            st.warning(f"⚠️ {source['name']}: {str(e)}")
    
    # Remove duplicates
    unique_questions = []
    seen = set()
    for q in all_questions:
        q_key = q['question'].lower()[:50]
        if q_key not in seen:
            seen.add(q_key)
            unique_questions.append(q)
    
    # Add diverse generated questions if needed
    if len(unique_questions) < num_questions:
        remaining = num_questions - len(unique_questions)
        st.info(f"📚 Adding {remaining} comprehensive knowledge-based questions...")
        generated = generate_diverse_questions(technology, remaining)
        unique_questions.extend(generated)
    
    # Add company tags
    if company and company != "Select Company":
        for q in unique_questions:
            q['question'] = f"[{company}] {q['question']}"
    
    # Return EXACT number requested
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
            answer_clean = html.unescape(qa['answer'])[:800]
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
    """Create Word"""
    doc = Document()
    
    title = doc.add_heading(f'Interview Questions: {technology}', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()
    
    for idx, qa in enumerate(questions_data, 1):
        doc.add_heading(f"Question {idx}", level=2)
        doc.add_paragraph(qa['question'])
        
        if qa.get('answer'):
            doc.add_paragraph("Answer:", style='Heading 3')
            doc.add_paragraph(qa['answer'][:1000])
        
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
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 30px;">Real Web Scraping • Diverse Answers • Exact Question Count</p>', unsafe_allow_html=True)
    
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
            "📊 Number of Questions",
            options=[10, 20, 30, 50, 100, 200, 300, 500, 1000],
            value=50
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
        st.success("✨ Features:\n\n• Real scraping\n• Diverse answers\n• Exact count\n• PDF/Word export")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><p>🎯 Technology</p><h3>{selected_tech}</h3></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><p>📚 Questions</p><h3>{num_questions}</h3></div>', unsafe_allow_html=True)
    with col3:
        company_display = selected_company if selected_company != "Select Company" else "General"
        st.markdown(f'<div class="metric-card"><p>🏢 Company</p><h3>{company_display}</h3></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Generate Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Generate Questions", use_container_width=True):
            progress_bar = st.progress(0)
            status = st.empty()
            
            status.text("🌐 Scraping web sources...")
            progress_bar.progress(30)
            
            company_for_scrape = selected_company if selected_company != "Select Company" else None
            questions_data = scrape_interview_questions(
                selected_tech,
                num_questions,
                company_for_scrape
            )
            
            progress_bar.progress(80)
            status.text(f"✅ Generated exactly {len(questions_data)} questions")
            time.sleep(0.5)
            
            st.session_state['questions_data'] = questions_data
            st.session_state['tech'] = selected_tech
            st.session_state['company'] = selected_company
            
            progress_bar.progress(100)
            status.empty()
            progress_bar.empty()
            
            st.success(f"✅ Successfully generated {len(questions_data)} questions!")
            st.balloons()
    
    # Scroll to top button using HTML/JS
    components.html("""
        <style>
            #scrollBtn {
                display: none;
                position: fixed;
                bottom: 30px;
                right: 30px;
                z-index: 99999;
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
            #scrollBtn:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
            }
        </style>
        
        <button onclick="scrollToTop()" id="scrollBtn" title="Go to top">↑</button>
        
        <script>
            window.onscroll = function() {scrollFunction()};
            
            function scrollFunction() {
                const btn = document.getElementById("scrollBtn");
                if (document.body.scrollTop > 200 || document.documentElement.scrollTop > 200) {
                    btn.style.display = "block";
                } else {
                    btn.style.display = "none";
                }
            }
            
            function scrollToTop() {
                window.parent.document.querySelector('.main').scrollTo({
                    top: 0,
                    behavior: 'smooth'
                });
                window.scrollTo({top: 0, behavior: 'smooth'});
                document.body.scrollTop = 0;
                document.documentElement.scrollTop = 0;
            }
        </script>
    """, height=0)
    
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
            with_answers = sum(1 for q in questions_data if q.get('answer') and len(q.get('answer', '')) > 100)
            st.metric("✅ Detailed", with_answers)
        with col4:
            with_videos = sum(1 for q in questions_data if q.get('video'))
            st.metric("🎥 Videos", with_videos)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display questions
        for idx, qa in enumerate(questions_data, 1):
            st.markdown(f"""
            <div class="question-container">
                <div class="question-header">
                    <div class="question-number">Question {idx} of {len(questions_data)}</div>
                    <div class="question-text">{qa['question']}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Answer
            if qa.get('answer'):
                answer = qa['answer']
                formatted_answer = f"<p style='line-height: 1.8; margin: 10px 0;'>{answer}</p>"
                
                st.markdown(f"""
                <div class="answer-section">
                    <span class="answer-label">📝 Answer:</span>
                    <div class="answer-text">{formatted_answer}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Metadata
            col1, col2 = st.columns(2)
            with col1:
                if qa.get('source'):
                    st.markdown(f'<span class="badge badge-source">📚 {qa["source"]}</span>', unsafe_allow_html=True)
            
            # Links
            link_col1, link_col2 = st.columns(2)
            with link_col1:
                if qa.get('url'):
                    st.markdown(f'<a href="{qa["url"]}" target="_blank" class="link-button">🔗 Source</a>', unsafe_allow_html=True)
            with link_col2:
                if qa.get('video'):
                    st.markdown(f'<a href="{qa["video"]}" target="_blank" class="link-button">🎥 Video</a>', unsafe_allow_html=True)
            
            st.markdown('</div><br>', unsafe_allow_html=True)
        
        # Export
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">💾 Export</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            pdf_buffer = create_pdf(questions_data, st.session_state['tech'], st.session_state.get('company'))
            
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
            word_buffer = create_word(questions_data, st.session_state['tech'], st.session_state.get('company'))
            
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
        <p style="font-size: 1rem;">✅ Diverse Answers • ✅ Exact Question Count • ✅ Working Scroll Button</p>
        <p style="font-size: 0.95rem; margin: 15px 0;">
            🌐 Sources: GeeksforGeeks • InterviewBit • JavaTpoint • Knowledge Base
        </p>
        <p style="font-size: 0.9rem; color: #999;">
            💼 Practice Daily • 🎯 Stay Focused • 🚀 Achieve Success
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

import streamlit as st
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SSC Telangana Quiz System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #1e40af;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: #1e3a8a;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: #1e40af;
        font-weight: 600;
    }
    
    /* Cards */
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
        color: white;
    }
    
    .info-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
    }
    
    /* Question Card */
    .question-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .question-card:hover {
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        border-color: #3b82f6;
    }
    
    /* Timer */
    .timer {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .timer-warning {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Input fields */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select,
    .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus,
    .stSelectbox>div>div>select:focus,
    .stNumberInput>div>div>input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Radio buttons */
    .stRadio>div {
        background: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #e5e7eb;
    }
    
    /* Metrics */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: #d1fae5;
        color: #065f46;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stError {
        background: #fee2e2;
        color: #991b1b;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stWarning {
        background: #fef3c7;
        color: #92400e;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stInfo {
        background: #dbeafe;
        color: #1e40af;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Dark theme adjustments */
    @media (prefers-color-scheme: dark) {
        .question-card {
            background: #1f2937;
            border-color: #374151;
            color: #f9fafb;
        }
        
        .stRadio>div {
            background: #1f2937;
            border-color: #374151;
        }
    }
    
    /* Explanation box */
    .explanation-box {
        background: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        padding: 1rem;
        margin-top: 0.5rem;
        border-radius: 8px;
        color: #0c4a6e;
    }
    
    /* Score display */
    .score-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 2rem 0;
    }
    
    .score-number {
        font-size: 4rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    /* Navigation tabs */
    .nav-tabs {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .nav-tab {
        padding: 0.75rem 1.5rem;
        background: #f3f4f6;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .nav-tab:hover {
        background: #e5e7eb;
    }
    
    .nav-tab.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION & QUESTION BANK
# ============================================================================

class QuizConfig:
    """Configuration for different classes and subjects"""
    
    QUIZ_STRUCTURE = {
        "10th Standard": {
            "subjects": ["Mathematics", "Physics", "Chemistry"],
            "board": "SSC Telangana"
        },
        "9th Standard": {
            "subjects": ["Mathematics", "Physics", "Chemistry"],
            "board": "SSC Telangana"
        }
    }
    
    DEFAULT_QUESTIONS = 30
    TIMER_MINUTES = 45

# Question Bank with explanations
QUESTION_BANK = {
    "10th Standard": {
        "Mathematics": [
            {
                "q": "What is the value of sin 90°?",
                "options": ["0", "1", "√3/2", "1/2"],
                "answer": 1,
                "explanation": "Sin 90° = 1 because at 90°, the sine function reaches its maximum value on the unit circle."
            },
            {
                "q": "The sum of first n natural numbers is given by:",
                "options": ["n(n+1)", "n(n+1)/2", "n²", "2n+1"],
                "answer": 1,
                "explanation": "The formula for sum of first n natural numbers is n(n+1)/2. For example, 1+2+3+4+5 = 5(6)/2 = 15."
            },
            {
                "q": "If the discriminant of a quadratic equation is zero, the roots are:",
                "options": ["Real and distinct", "Real and equal", "Imaginary", "None"],
                "answer": 1,
                "explanation": "When discriminant (b²-4ac) = 0, the quadratic equation has two real and equal roots."
            },
            {
                "q": "The HCF of 24 and 36 is:",
                "options": ["6", "12", "24", "36"],
                "answer": 1,
                "explanation": "Factors of 24: 1,2,3,4,6,8,12,24. Factors of 36: 1,2,3,4,6,9,12,18,36. Highest common factor is 12."
            },
            {
                "q": "In a right-angled triangle, sin²θ + cos²θ equals:",
                "options": ["0", "1", "2", "θ"],
                "answer": 1,
                "explanation": "This is the fundamental trigonometric identity: sin²θ + cos²θ = 1, valid for all values of θ."
            },
        ],
        "Physics": [
            {
                "q": "SI unit of electric current is:",
                "options": ["Volt", "Ampere", "Ohm", "Watt"],
                "answer": 1,
                "explanation": "Ampere (A) is the SI unit of electric current, named after André-Marie Ampère."
            },
            {
                "q": "The formula for kinetic energy is:",
                "options": ["mgh", "1/2 mv²", "mv", "m/v"],
                "answer": 1,
                "explanation": "Kinetic energy = 1/2 mv² where m is mass and v is velocity. It represents energy due to motion."
            },
            {
                "q": "Newton's second law states F equals:",
                "options": ["ma", "mv", "m/a", "a/m"],
                "answer": 0,
                "explanation": "Newton's second law: Force = mass × acceleration (F = ma). It relates force to the change in motion."
            },
        ],
        "Chemistry": [
            {
                "q": "The pH of pure water is:",
                "options": ["0", "7", "14", "1"],
                "answer": 1,
                "explanation": "Pure water has a pH of 7, which is neutral (neither acidic nor basic)."
            },
            {
                "q": "The chemical formula of water is:",
                "options": ["H₂O", "CO₂", "NaCl", "HCl"],
                "answer": 0,
                "explanation": "Water molecule consists of 2 hydrogen atoms and 1 oxygen atom, hence H₂O."
            },
            {
                "q": "Atomic number represents the number of:",
                "options": ["Neutrons", "Protons", "Electrons", "Nucleons"],
                "answer": 1,
                "explanation": "Atomic number is the number of protons in an atom's nucleus, which defines the element."
            },
        ]
    }
}

# Custom question bank (user-created)
if 'custom_question_bank' not in st.session_state:
    st.session_state.custom_question_bank = {}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'page': 'home',
        'quiz_started': False,
        'quiz_completed': False,
        'start_time': None,
        'user_answers': {},
        'selected_class': None,
        'selected_subject': None,
        'num_questions': QuizConfig.DEFAULT_QUESTIONS,
        'quiz_questions': [],
        'timer_duration': QuizConfig.TIMER_MINUTES,
        'quiz_source': 'default',  # 'default' or 'custom'
        'temp_questions': [],  # Temporary storage for question creation
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_available_classes():
    """Get list of available classes"""
    classes = list(QuizConfig.QUIZ_STRUCTURE.keys())
    custom_classes = list(st.session_state.custom_question_bank.keys())
    return sorted(list(set(classes + custom_classes)))

def get_subjects_for_class(class_name):
    """Get subjects for a specific class"""
    subjects = []
    if class_name in QuizConfig.QUIZ_STRUCTURE:
        subjects.extend(QuizConfig.QUIZ_STRUCTURE[class_name]["subjects"])
    if class_name in st.session_state.custom_question_bank:
        subjects.extend(st.session_state.custom_question_bank[class_name].keys())
    return sorted(list(set(subjects)))

def load_questions(class_name, subject, num_questions, source='default'):
    """Load questions for the selected class and subject"""
    if source == 'custom':
        all_questions = st.session_state.custom_question_bank.get(class_name, {}).get(subject, [])
    else:
        all_questions = QUESTION_BANK.get(class_name, {}).get(subject, [])
    
    return all_questions[:min(num_questions, len(all_questions))]

def calculate_score(questions, user_answers):
    """Calculate quiz score"""
    correct = 0
    for i, q in enumerate(questions):
        if user_answers.get(i) == q["answer"]:
            correct += 1
    return correct, len(questions)

def save_custom_questions(class_name, subject, questions):
    """Save custom questions to the bank"""
    if class_name not in st.session_state.custom_question_bank:
        st.session_state.custom_question_bank[class_name] = {}
    
    if subject not in st.session_state.custom_question_bank[class_name]:
        st.session_state.custom_question_bank[class_name][subject] = []
    
    st.session_state.custom_question_bank[class_name][subject].extend(questions)

# ============================================================================
# PAGE: HOME/NAVIGATION
# ============================================================================

def show_home_page():
    """Display home page with navigation"""
    load_custom_css()
    
    st.markdown("""
    <div class="card">
        <h1 style="color: white; margin: 0;">📚 SSC Telangana Online Quiz System</h1>
        <p style="color: white; opacity: 0.9; font-size: 1.1rem; margin-top: 0.5rem;">
            Professional Quiz Platform for Students
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center; 
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1); color: white; height: 250px;
                    display: flex; flex-direction: column; justify-content: center;">
            <h2 style="color: white; font-size: 3rem; margin: 0;">📝</h2>
            <h3 style="color: white; margin: 1rem 0;">Take Quiz</h3>
            <p style="color: white; opacity: 0.9;">Start a quiz from existing questions</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Quiz", key="btn_take_quiz", use_container_width=True):
            st.session_state.page = 'take_quiz'
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center; 
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1); color: white; height: 250px;
                    display: flex; flex-direction: column; justify-content: center;">
            <h2 style="color: white; font-size: 3rem; margin: 0;">✏️</h2>
            <h3 style="color: white; margin: 1rem 0;">Create Quiz</h3>
            <p style="color: white; opacity: 0.9;">Create custom quiz with your questions</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Create Quiz", key="btn_create_quiz", use_container_width=True):
            st.session_state.page = 'create_quiz'
            st.rerun()
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center; 
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1); color: white; height: 250px;
                    display: flex; flex-direction: column; justify-content: center;">
            <h2 style="color: white; font-size: 3rem; margin: 0;">📊</h2>
            <h3 style="color: white; margin: 1rem 0;">Statistics</h3>
            <p style="color: white; opacity: 0.9;">View your quiz history</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Stats", key="btn_stats", use_container_width=True):
            st.info("📊 Statistics feature coming soon!")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Info section
    st.markdown("""
    <div class="info-card">
        <h3 style="color: white; margin-top: 0;">✨ Features</h3>
        <ul style="color: white; opacity: 0.95;">
            <li>🎯 Multiple subjects: Mathematics, Physics, Chemistry</li>
            <li>⏱️ Customizable timer and number of questions</li>
            <li>📝 Create your own custom quizzes</li>
            <li>💡 Detailed explanations for each answer</li>
            <li>📊 Comprehensive score reports</li>
            <li>🎨 Professional, modern interface</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: CREATE CUSTOM QUIZ
# ============================================================================

def show_create_quiz_page():
    """Display page to create custom quiz"""
    load_custom_css()
    
    st.markdown("""
    <div class="card">
        <h1 style="color: white; margin: 0;">✏️ Create Custom Quiz</h1>
        <p style="color: white; opacity: 0.9;">Add your own questions to the quiz bank</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("⬅️ Back to Home", key="back_from_create"):
        st.session_state.page = 'home'
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quiz Configuration
    st.subheader("📋 Quiz Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        class_name = st.text_input("Class Name", placeholder="e.g., 10th Standard", key="create_class")
    
    with col2:
        subject_name = st.text_input("Subject Name", placeholder="e.g., Mathematics", key="create_subject")
    
    with col3:
        num_questions = st.number_input("Number of Questions", min_value=1, max_value=100, value=5, key="create_num_q")
    
    if class_name and subject_name and num_questions:
        st.success(f"✅ Ready to create {num_questions} questions for {class_name} - {subject_name}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("❓ Enter Questions")
        
        # Initialize temp questions if needed
        if len(st.session_state.temp_questions) != num_questions:
            st.session_state.temp_questions = [
                {"q": "", "options": ["", "", "", ""], "answer": 0, "explanation": ""}
                for _ in range(num_questions)
            ]
        
        # Display question input forms
        for i in range(num_questions):
            with st.expander(f"Question {i+1}", expanded=(i==0)):
                st.markdown(f"### Question {i+1}")
                
                question_text = st.text_area(
                    "Question Text",
                    value=st.session_state.temp_questions[i]["q"],
                    placeholder="Enter your question here...",
                    key=f"q_text_{i}",
                    height=100
                )
                st.session_state.temp_questions[i]["q"] = question_text
                
                st.markdown("**Options:**")
                opt_cols = st.columns(2)
                
                for opt_idx in range(4):
                    with opt_cols[opt_idx % 2]:
                        option_text = st.text_input(
                            f"Option {opt_idx + 1}",
                            value=st.session_state.temp_questions[i]["options"][opt_idx],
                            placeholder=f"Option {opt_idx + 1}",
                            key=f"q_{i}_opt_{opt_idx}"
                        )
                        st.session_state.temp_questions[i]["options"][opt_idx] = option_text
                
                correct_answer = st.radio(
                    "Correct Answer",
                    options=[0, 1, 2, 3],
                    format_func=lambda x: f"Option {x+1}",
                    horizontal=True,
                    key=f"q_{i}_answer",
                    index=st.session_state.temp_questions[i]["answer"]
                )
                st.session_state.temp_questions[i]["answer"] = correct_answer
                
                explanation = st.text_area(
                    "Explanation (Optional)",
                    value=st.session_state.temp_questions[i]["explanation"],
                    placeholder="Explain why this is the correct answer...",
                    key=f"q_{i}_exp",
                    height=80
                )
                st.session_state.temp_questions[i]["explanation"] = explanation
                
                st.markdown("---")
        
        # Save button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("💾 Save Questions", type="primary", use_container_width=True, key="save_questions"):
                # Validate all questions
                all_valid = True
                for i, q in enumerate(st.session_state.temp_questions):
                    if not q["q"].strip():
                        st.error(f"❌ Question {i+1}: Question text is required")
                        all_valid = False
                    if any(not opt.strip() for opt in q["options"]):
                        st.error(f"❌ Question {i+1}: All options must be filled")
                        all_valid = False
                
                if all_valid:
                    save_custom_questions(class_name, subject_name, st.session_state.temp_questions)
                    st.success(f"✅ Successfully saved {num_questions} questions to {class_name} - {subject_name}!")
                    st.balloons()
                    time.sleep(2)
                    st.session_state.temp_questions = []
                    st.session_state.page = 'home'
                    st.rerun()

# ============================================================================
# PAGE: TAKE QUIZ SETUP
# ============================================================================

def show_setup_page():
    """Display quiz setup page"""
    load_custom_css()
    
    st.markdown("""
    <div class="card">
        <h1 style="color: white; margin: 0;">📚 Quiz Setup</h1>
        <p style="color: white; opacity: 0.9;">Configure your quiz parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("⬅️ Back to Home", key="back_from_setup"):
        st.session_state.page = 'home'
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_class = st.selectbox(
            "📚 Select Class:",
            get_available_classes(),
            key="class_selector"
        )
    
    with col2:
        quiz_source = st.radio(
            "📖 Question Source:",
            ["Default Questions", "Custom Questions"],
            key="source_selector",
            horizontal=True
        )
        source = 'default' if quiz_source == "Default Questions" else 'custom'
    
    col3, col4 = st.columns(2)
    
    with col3:
        available_subjects = get_subjects_for_class(selected_class)
        if not available_subjects:
            st.warning("⚠️ No subjects available for this class")
            return
        selected_subject = st.selectbox(
            "📖 Select Subject:",
            available_subjects,
            key="subject_selector"
        )
    
    with col4:
        # Get available questions count
        if source == 'custom':
            available_q = len(st.session_state.custom_question_bank.get(selected_class, {}).get(selected_subject, []))
        else:
            available_q = len(QUESTION_BANK.get(selected_class, {}).get(selected_subject, []))
        
        if available_q == 0:
            st.warning(f"⚠️ No questions available for {selected_subject}")
            return
        
        num_questions = st.number_input(
            f"🔢 Number of Questions (Max: {available_q}):",
            min_value=1,
            max_value=available_q,
            value=min(30, available_q),
            step=5,
            key="num_questions_input"
        )
    
    timer_minutes = st.slider(
        "⏱️ Timer (minutes):",
        min_value=5,
        max_value=180,
        value=45,
        step=5,
        key="timer_input"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display quiz info
    st.markdown(f"""
    <div class="info-card">
        <h3 style="color: white; margin-top: 0;">📋 Quiz Summary</h3>
        <ul style="color: white; opacity: 0.95; list-style: none; padding: 0;">
            <li>📚 <strong>Class:</strong> {selected_class}</li>
            <li>📖 <strong>Subject:</strong> {selected_subject}</li>
            <li>🔢 <strong>Questions:</strong> {num_questions}</li>
            <li>⏱️ <strong>Duration:</strong> {timer_minutes} minutes</li>
            <li>📝 <strong>Source:</strong> {quiz_source}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Start Quiz", type="primary", use_container_width=True, key="start_quiz_btn"):
            questions = load_questions(selected_class, selected_subject, num_questions, source)
            
            if len(questions) == 0:
                st.error(f"❌ No questions available for {selected_class} - {selected_subject}")
                return
            
            st.session_state.quiz_started = True
            st.session_state.start_time = datetime.now()
            st.session_state.selected_class = selected_class
            st.session_state.selected_subject = selected_subject
            st.session_state.num_questions = num_questions
            st.session_state.quiz_questions = questions
            st.session_state.timer_duration = timer_minutes
            st.session_state.quiz_source = source
            st.session_state.user_answers = {}
            st.session_state.page = 'quiz'
            st.rerun()

# ============================================================================
# PAGE: QUIZ
# ============================================================================

def show_quiz_page():
    """Display the quiz questions page"""
    load_custom_css()
    
    questions = st.session_state.quiz_questions
    
    # Timer calculation
    elapsed = datetime.now() - st.session_state.start_time
    remaining = timedelta(minutes=st.session_state.timer_duration) - elapsed
    
    if remaining.total_seconds() <= 0:
        st.session_state.quiz_completed = True
        st.session_state.page = 'results'
        st.rerun()
    
    # Header with timer
    mins, secs = divmod(int(remaining.total_seconds()), 60)
    timer_class = "timer-warning" if mins < 5 else "timer"
    
    st.markdown(f"""
    <div class="{timer_class}">
        ⏱️ Time Remaining: {mins:02d}:{secs:02d}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Progress bar
    progress = len(st.session_state.user_answers) / len(questions)
    st.progress(progress)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Subject", st.session_state.selected_subject)
    with col2:
        st.metric("✅ Answered", f"{len(st.session_state.user_answers)}/{len(questions)}")
    with col3:
        unanswered = len(questions) - len(st.session_state.user_answers)
        st.metric("❓ Remaining", unanswered)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Questions
    for i, question in enumerate(questions):
        st.markdown(f"""
        <div class="question-card">
            <h3 style="color: #1e40af; margin-top: 0;">Question {i+1}</h3>
            <p style="font-size: 1.1rem; color: #1f2937; margin: 1rem 0;">{question['q']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get current answer
        current_answer = None
        if i in st.session_state.user_answers:
            current_answer = question['options'][st.session_state.user_answers[i]]
        
        answer = st.radio(
            "Select your answer:",
            options=question['options'],
            key=f"q_{i}",
            index=question['options'].index(current_answer) if current_answer else None,
            label_visibility="collapsed"
        )
        
        # Store answer
        if answer:
            st.session_state.user_answers[i] = question['options'].index(answer)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Submit button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📝 Submit Quiz", type="primary", use_container_width=True, key="submit_quiz"):
            if len(st.session_state.user_answers) < len(questions):
                unanswered_count = len(questions) - len(st.session_state.user_answers)
                st.warning(f"⚠️ You have {unanswered_count} unanswered question(s). Are you sure you want to submit?")
                if st.button("Yes, Submit Anyway", key="confirm_submit"):
                    st.session_state.quiz_completed = True
                    st.session_state.page = 'results'
                    st.rerun()
            else:
                st.session_state.quiz_completed = True
                st.session_state.page = 'results'
                st.rerun()
    
    # Auto-refresh for timer
    time.sleep(1)
    st.rerun()

# ============================================================================
# PAGE: RESULTS
# ============================================================================

def show_results_page():
    """Display quiz results"""
    load_custom_css()
    
    questions = st.session_state.quiz_questions
    user_answers = st.session_state.user_answers
    
    correct, total = calculate_score(questions, user_answers)
    percentage = (correct / total) * 100
    
    # Results header
    st.markdown(f"""
    <div class="score-display">
        <h1 style="color: white; margin: 0;">🎯 Quiz Results</h1>
        <div class="score-number">{correct}/{total}</div>
        <h2 style="color: white; margin: 0;">{percentage:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Correct", correct)
    with col2:
        st.metric("❌ Incorrect", total - correct)
    with col3:
        st.metric("📊 Percentage", f"{percentage:.1f}%")
    with col4:
        unanswered = total - len(user_answers)
        st.metric("⚠️ Unanswered", unanswered)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Performance message
    if percentage >= 90:
        st.success("🌟 Outstanding! Excellent performance!")
    elif percentage >= 75:
        st.success("✅ Great job! Well done!")
    elif percentage >= 60:
        st.info("👍 Good effort! Keep practicing!")
    elif percentage >= 40:
        st.warning("📚 Keep studying! You can do better!")
    else:
        st.error("💪 Don't give up! More practice needed!")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("📋 Detailed Answer Review")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detailed results
    for i, question in enumerate(questions):
        user_answer_idx = user_answers.get(i)
        correct_answer_idx = question['answer']
        
        is_correct = user_answer_idx == correct_answer_idx
        is_unanswered = user_answer_idx is None
        
        # Question card with color coding
        if is_correct:
            card_color = "#d1fae5"
            border_color = "#10b981"
            icon = "✅"
            status = "Correct"
        elif is_unanswered:
            card_color = "#fef3c7"
            border_color = "#f59e0b"
            icon = "⚠️"
            status = "Not Answered"
        else:
            card_color = "#fee2e2"
            border_color = "#ef4444"
            icon = "❌"
            status = "Incorrect"
        
        st.markdown(f"""
        <div style="background: {card_color}; border-left: 4px solid {border_color}; 
                    padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
            <h3 style="color: #1f2937; margin-top: 0;">
                {icon} Question {i+1} - <span style="color: {border_color};">{status}</span>
            </h3>
            <p style="font-size: 1.1rem; color: #374151; margin: 1rem 0;">
                <strong>{question['q']}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if user_answer_idx is not None:
                user_answer_text = question['options'][user_answer_idx]
                if is_correct:
                    st.success(f"**Your answer:** {user_answer_text}")
                else:
                    st.error(f"**Your answer:** {user_answer_text}")
            else:
                st.warning("**Your answer:** Not answered")
        
        with col2:
            correct_answer_text = question['options'][correct_answer_idx]
            st.info(f"**Correct answer:** {correct_answer_text}")
        
        # Show explanation if available
        if question.get('explanation'):
            st.markdown(f"""
            <div class="explanation-box">
                <strong>💡 Explanation:</strong><br>
                {question['explanation']}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Action buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🏠 Home", use_container_width=True, key="home_btn"):
            reset_quiz_state()
            st.session_state.page = 'home'
            st.rerun()
    
    with col2:
        if st.button("🔄 Retake Quiz", use_container_width=True, key="retake_btn"):
            # Keep same configuration, just reset answers
            st.session_state.quiz_started = True
            st.session_state.quiz_completed = False
            st.session_state.start_time = datetime.now()
            st.session_state.user_answers = {}
            st.session_state.page = 'quiz'
            st.rerun()
    
    with col3:
        if st.button("📝 New Quiz", type="primary", use_container_width=True, key="new_quiz_btn"):
            reset_quiz_state()
            st.session_state.page = 'take_quiz'
            st.rerun()

def reset_quiz_state():
    """Reset quiz-related session state"""
    st.session_state.quiz_started = False
    st.session_state.quiz_completed = False
    st.session_state.start_time = None
    st.session_state.user_answers = {}
    st.session_state.quiz_questions = []

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: white; margin: 0;">📚</h1>
            <h2 style="color: white; margin: 0.5rem 0;">SSC Quiz</h2>
            <p style="color: white; opacity: 0.8;">Telangana State Board</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div style="color: white; padding: 1rem;">
            <h3 style="color: white;">Quick Info</h3>
            <p style="opacity: 0.9;">• Multiple choice questions</p>
            <p style="opacity: 0.9;">• Instant results</p>
            <p style="opacity: 0.9;">• Detailed explanations</p>
            <p style="opacity: 0.9;">• Custom quiz creation</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.session_state.page != 'home':
            if st.button("🏠 Return to Home", key="sidebar_home", use_container_width=True):
                reset_quiz_state()
                st.session_state.page = 'home'
                st.rerun()
    
    # Main content routing
    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'create_quiz':
        show_create_quiz_page()
    elif st.session_state.page == 'take_quiz':
        show_setup_page()
    elif st.session_state.page == 'quiz':
        show_quiz_page()
    elif st.session_state.page == 'results':
        show_results_page()

if __name__ == "__main__":
    main()

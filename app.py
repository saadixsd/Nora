"""
Nora AI - Canadian Legal Assistant Flask Application
A Flask-based AI legal assistant for Canadian law queries and document analysis
"""

import os
import sys
import time
import logging
import re
from functools import lru_cache

# Flask and web-related imports
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

# Data processing imports
import pandas as pd
from datasets import load_dataset

# AI and ML imports
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Document processing imports
from PyPDF2 import PdfReader
import docx

# ==================== CONFIGURATION ====================

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Context limits
MAX_CONTEXT_CHARACTERS = 3000
MAX_CONTEXT_LINES = 50
MAX_USER_INPUT_LENGTH = 1000
MAX_QUERY_LENGTH = 100
MAX_CASE_RESULTS = 100

# AI model configuration
AI_MODEL_NAME = "llama3.2"
AI_RETRY_ATTEMPTS = 3
AI_RETRY_DELAY = 2

# ==================== LOGGING SETUP ====================

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ==================== FLASK APP INITIALIZATION ====================

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# CORS configuration
CORS(app, resources={
    r"/*": {"origins": ["https://www.xenoraai.com", "https://xenoraai.com"]}
})

# ==================== SECURITY HEADERS ====================

@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# ==================== AI MODEL SETUP ====================

model = OllamaLLM(model=AI_MODEL_NAME, timeout=None)

template = """
Hi there! I'm Nora, your AI legal assistant specializing in Canadian law. 
Whether you're a practicing lawyer or a student navigating your studies, 
I'm here to provide reliable insights, research assistance, and support for your legal tasks.

Feel free to ask me about case law, legislation, or practical tips for handling legal matters. 

**Conversation History:**
{context}
User: {question}
Nora:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# ==================== ROLE CONFIGURATION ====================

role_messages = {
    "lawyer": "I can assist with legal case analysis, referencing precedents, or general legal queries. Let me know how I can assist",
    "student": "I can help with assignments, legal concepts, or academic guidance. Let me know how I can assist",
    "user": "Feel free to ask me anything, whether it's general queries or specific tasks. Let me know how I can assist",
}

# ==================== GLOBAL VARIABLES ====================

conversation_history = []

# ==================== UTILITY FUNCTIONS ====================

@lru_cache(maxsize=100)
def extract_user_details(introduction_text):
    """Extract user's name and role from introduction text."""
    if not isinstance(introduction_text, str):
        return "User", "user"
    
    name_patterns = [
        r"(?i)(?:my name is|i'm|i am|this is)\s+([A-Za-z]+)",
        r"(?i)(?:hi,?\s*|hello,?\s*)i'm\s+([A-Za-z]+)",
        r"(?i)([A-Za-z]+)\s+a\s+(lawyer|lawstudent|student|academic)"
    ]
    
    role_patterns = [
        r"(?i)\b(?:lawyer|legal professional|attorney|law student|student|academic|paralegal|professor)\b"
    ]

    name = "User"
    role = "user"

    try:
        # Extract name
        for pattern in name_patterns:
            match = re.search(pattern, introduction_text)
            if match:
                potential_name = match.group(1).capitalize()
                name = potential_name if len(potential_name) <= 50 else "User"
                break

        # Extract role
        for pattern in role_patterns:
            match = re.search(pattern, introduction_text)
            if match:
                role = match.group(0).lower()
                break
                
    except Exception as e:
        logger.error(f"Error in name extraction: {e}")
        return "User", "user"

    return name, role

def truncate_context(context, max_characters=MAX_CONTEXT_CHARACTERS, max_lines=MAX_CONTEXT_LINES):
    """Truncate context to stay within limits."""
    if not context:
        return ""
    
    try:
        lines = context.strip().split("\n")
        lines = lines[-max_lines:]
        truncated_context = "\n".join(lines)
        
        if len(truncated_context) > max_characters:
            truncated_context = truncated_context[-max_characters:]
        
        return truncated_context
    except Exception as e:
        logger.error(f"Error truncating context: {e}")
        return ""

def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def print_real_time(text, delay=0.015, max_length=500):
    """Print text with real-time typing effect for CLI mode."""
    try:
        if len(text) > max_length:
            print(text)
            return
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        print()
    except Exception:
        print(text)

# ==================== DOCUMENT PROCESSING ====================

def extract_text_from_file(filepath):
    """Extract text from uploaded files (PDF, DOCX, TXT)."""
    ext = filepath.rsplit('.', 1)[-1].lower()

    try:
        if ext == 'pdf':
            with open(filepath, 'rb') as f:
                reader = PdfReader(f)
                text = []
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text.append(extracted)
                return ' '.join(text)

        elif ext == 'docx':
            doc = docx.Document(filepath)
            return '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])

        elif ext == 'txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()

    except Exception as e:
        logger.error(f"Error extracting text from {filepath}: {e}")
        return ''

    return ''

# ==================== AI CONVERSATION HANDLING ====================

def handle_conversation(user_input, context, user_role):
    """Handle user conversations with input validation and error handling."""
    if not user_input or len(user_input) > MAX_USER_INPUT_LENGTH:
        return "Please provide a valid message under 1000 characters."
    
    try:
        global conversation_history
        
        # Add greeting for first message
        greeting = ""
        if not conversation_history:
            greeting = "Hello! I'm Nora, your AI legal assistant. I'm ready to help you with legal questions and provide data visualizations. What would you like to know?\n\n"
        
        conversation_history.append(user_input)
        
        # Extract user details
        user_name, role = extract_user_details(user_input)
        role_message = role_messages.get(user_role, "How can I assist you today?")
        
        # Generate AI response with retry mechanism
        ai_response = None
        for attempt in range(AI_RETRY_ATTEMPTS):
            try:
                ai_response = chain.invoke({
                    "context": context,
                    "question": greeting + user_input
                }).strip()
                break
            except Exception as e:
                logger.warning(f"AI response attempt {attempt + 1} failed: {e}")
                if attempt == AI_RETRY_ATTEMPTS - 1:
                    logger.error(f"AI model failed after {AI_RETRY_ATTEMPTS} attempts")
                    return "I'm sorry, I'm having trouble processing that right now. Please try again."
                time.sleep(AI_RETRY_DELAY)

        return ai_response
    
    except Exception as e:
        logger.error(f"Error in conversation handling: {e}")
        return "I'm sorry, something went wrong. Could you rephrase that?"

# ==================== CASE DATA FUNCTIONS ====================

@lru_cache(maxsize=50)
def get_case_data(query):
    """Retrieve case data with caching and error handling."""
    if not query or len(query) > MAX_QUERY_LENGTH:
        return "Please provide a valid search query under 100 characters."
    
    try:
        dataset = load_dataset("refugee-law-lab/canadian-legal-data", split="train")
        case_data = {'Case Name': [], 'Case Type': [], 'Citation': []}
        
        for entry in dataset:
            if len(case_data['Case Name']) >= MAX_CASE_RESULTS:
                break
                
            case_name = entry.get("case_name", "Unknown")
            case_type = entry.get("case_type", "Unknown")
            citation = entry.get("citation", "Unknown")
            
            if query.lower() in case_name.lower() or query.lower() in case_type.lower():
                case_data['Case Name'].append(case_name)
                case_data['Case Type'].append(case_type)
                case_data['Citation'].append(citation)
        
        return pd.DataFrame(case_data) if case_data['Case Name'] else None
    
    except Exception as e:
        logger.error(f"Error in case data retrieval: {e}")
        return None

def recommend_cases(query, case_data):
    """Recommend similar cases based on query."""
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(case_data['Case Name'])
        query_vector = vectorizer.transform([query])
        similarity = cosine_similarity(query_vector, tfidf_matrix)
        top_indices = similarity.argsort()[0, -5:][::-1]
        return case_data.iloc[top_indices]
    except Exception as e:
        logger.error(f"Error in case recommendation: {e}")
        return case_data.head(5)

# ==================== FLASK ROUTES ====================

@app.route('/')
def home():
    """Render the main application page."""
    return render_template('nora.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'ok'}), 200

@app.route('/ask', methods=['POST'])
def ask():
    """Handle user questions."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400
            
        question = data.get('message', '').strip()
        if not question:
            return jsonify({'error': 'Message is required'}), 400
            
        context = data.get('context', '')
        response = handle_conversation(question, context, 'user')
        
        return jsonify({
            'answer': response,
            'context': context
        })
    
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads."""
    if 'file' not in request.files:
        logger.warning("Upload attempt with no file part")
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("Upload attempt with empty filename")
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"File uploaded successfully: {filename}")
        
        text = extract_text_from_file(filepath)
        response = handle_conversation(text[:1000], context="", user_role='user')
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except Exception as e:
            logger.warning(f"Could not remove uploaded file: {e}")
        
        return jsonify({
            'filename': filename,
            'answer': response
        })

    logger.warning(f"Unsupported file type upload attempt: {file.filename}")
    return jsonify({'error': 'Unsupported file type'}), 400

@app.route('/analyze-document', methods=['POST'])
def analyze_document():
    """Analyze uploaded documents."""
    file = request.files.get('file')
    document_type = request.form.get('documentType')
    analysis_type = request.form.get('analysisType')
    
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    return jsonify({
        'status': 'success',
        'document_type': document_type,
        'analysis_type': analysis_type,
        'analysis_summary': "This sample contract contains 15 clauses, with 3 potential risk areas identified.",
        'risk_areas': [
            "Indemnification Clause - Unusually broad language",
            "Termination Section - Notice period shorter than standard",
            "Governing Law - Specifies foreign jurisdiction"
        ]
    })

@app.route('/ask-question', methods=['POST'])
def ask_question():
    """Handle structured legal questions."""
    data = request.get_json()
    question = data.get('question')
    jurisdiction = data.get('jurisdiction')
    complexity = data.get('complexity')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    context = ""
    answer = handle_conversation(question, context, 'user')
    
    return jsonify({
        'status': 'success',
        'answer': answer
    })

# ==================== CLI INTERFACE ====================

def web_interaction():
    """Handle command-line interaction with Nora."""
    print("\nNora: Hello! I'm Nora, your AI legal assistant. I'm ready to help you with legal questions and provide data visualizations. What would you like to know?\n")
    
    context = ""
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("\nNora: Goodbye! Have a great day!\n")
                break

            user_name, user_role = extract_user_details(user_input)
            response = handle_conversation(user_input, context, user_role)
            
            print_real_time(f"\nNora: {response}\n")
            
            context += f"\nUser: {user_input}\nNora: {response}"
            context = truncate_context(context)
        
        except KeyboardInterrupt:
            print("\nNora: Goodbye! Have a great day!\n")
            break
        
        except Exception as e:
            logger.error(f"Error in web interaction: {e}")
            print("\nNora: I encountered an error. Please try again.\n")

# ==================== APPLICATION ENTRY POINT ====================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug_mode
    )

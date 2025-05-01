from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import re
import os
import time
import sys
import logging
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from datasets import load_dataset
from functools import lru_cache
from werkzeug.middleware.proxy_fix import ProxyFix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import docx

# Initialize Flask application
app = Flask(__name__)

# Add proxy support for handling requests through proxies (e.g., load balancers)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Enable Cross-Origin Resource Sharing (CORS) for specific origins
CORS(app, resources={r"/*": {"origins": ["https://www.xenoraai.com", "http://localhost:5000", "http://127.0.0.1:5000"]}})

logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppresses INFO logs from httpx
logging.basicConfig(level=logging.WARNING)  # Show only warnings and errors

# Get logger instance
logger = logging.getLogger(__name__)

# Initialize the AI model (e.g., LLaMA model) with a timeout for requests
# The model is used to handle legal queries and provide insights
model = OllamaLLM(model="llama3.2", timeout=None)

# Define a template for generating responses, which includes a conversation history and question
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

# Create a chat prompt from the template, which is then used for generating the response
prompt = ChatPromptTemplate.from_template(template)

# Chain the prompt with the AI model to handle conversation processing
chain = prompt | model

# Define role-specific messages that provide context for different user roles (lawyer, student, general user)
role_messages = {
    "lawyer": "I can assist with legal case analysis, referencing precedents, or general legal queries. Let me know how I can assist",
    "student": "I can help with assignments, legal concepts, or academic guidance. Let me know how I can assist",
    "user": "Feel free to ask me anything, whether it's general queries or specific tasks. Let me know how I can assist",
}

# Define the function to extract user's name and role from an introduction, with caching
@lru_cache(maxsize=100)
def extract_user_details(introduction_text):
    """Extracts the user's name and role from their introduction with caching."""
    
    # Return default values if the input is not a string
    if not isinstance(introduction_text, str):
        return "User", "user"
    
    # Define patterns to match possible names in the introduction
    name_patterns = [
        r"(?i)(?:my name is|i'm|i am|this is)\s+([A-Za-z]+)",  # Matches common phrases like "My name is"
        r"(?i)(?:hi,?\s*|hello,?\s*)i'm\s+([A-Za-z]+)",          # Matches "Hi, I'm [name]"
        r"(?i)([A-Za-z]+)\s+a\s+(lawyer|lawstudent|student|academic)"  # Matches "[Name] a [role]"
    ]
    
    # Define pattern to match roles (e.g., lawyer, student)
    role_patterns = [
    r"(?i)\b(?:lawyer|legal professional|attorney|law student|student|academic|paralegal|professor)\b"
    ]

    name = "User"  # Default name
    role = "user"  # Default role

    try:
        # Try to extract name from the introduction text using predefined patterns
        for pattern in name_patterns:
            match = re.search(pattern, introduction_text)
            if match:
                potential_name = match.group(1).capitalize()  # Capitalize the name
                name = potential_name if len(potential_name) <= 50 else "User"  # Ensure name is not too long
                break

        # Try to extract role from the introduction text using predefined patterns
        for pattern in role_patterns:
            match = re.search(pattern, introduction_text)
            if match:
                role = match.group(0).lower()  # Convert role to lowercase
                break
    except Exception as e:
        # Log any error during the extraction process
        logger.error(f"Error in name extraction: {e}")
        return "User", "user"  # Return default values in case of error

    # Return the extracted name and role
    return name, role


# Define the function to truncate context based on line and character limits
def truncate_context(context, max_characters=3000, max_lines=50):
    """Truncate context to stay within max characters and lines."""
    if not context:
        return ""
    try:
        # Split into lines
        lines = context.strip().split("\n")
        # Keep only the last `max_lines`
        lines = lines[-max_lines:]
        truncated_context = "\n".join(lines)
        
        # Further truncate by characters
        if len(truncated_context) > max_characters:
            truncated_context = truncated_context[-max_characters:]
        
        return truncated_context
    except Exception as e:
        logger.error(f"Error truncating context: {e}")
        return ""

        

# Define a global variable to track conversation history
conversation_history = []

# Define the function to handle user conversations
def handle_conversation(user_input, context, user_role):
    """Enhanced conversation handler with input validation, error handling, and natural greetings."""
    
    # Check if the user input is valid (non-empty and under 1000 characters)
    if not user_input or len(user_input) > 1000:
        return "Please provide a valid message under 1000 characters."
    
    try:
        # First-message greeting logic
        global conversation_history
        if not conversation_history:
            # It's the start of a new conversation
            greeting = "Hello! I'm Nora, your AI legal assistant. I'm ready to help you with legal questions and provide data visualizations. What would you like to know?\n\n"
        else:
            greeting = ""

        conversation_history.append(user_input)  # Save user input into history

        # Extract user details (e.g., name and role) from the input
        user_name, role = extract_user_details(user_input)
        
        # Get a role-specific message based on the user's role
        role_message = role_messages.get(user_role, "How can I assist you today?")
        
        # Add timeout handling for AI response with retry mechanism
        ai_response = None
        retry_attempts = 3  # Set the number of retry attempts
        for attempt in range(retry_attempts):
            try:
                # Try invoking the AI model to generate a response
                ai_response = chain.invoke({
                    "context": context,
                    "question": greeting + user_input  # Add greeting if it's first message
                }).strip()  # Strip any leading/trailing whitespace
                break  # Exit the loop if the response is successful
            except Exception as e:
                # Log a warning if the AI response attempt fails
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == retry_attempts - 1:
                    # Log error if all attempts fail
                    logger.error(f"AI model failed after {retry_attempts} attempts")
                    return "I'm sorry, I'm having trouble processing that right now. Please try again."
                time.sleep(2)  # Wait before retrying

        return f"{ai_response}"
    
    except Exception as e:
        logger.error(f"Error in conversation handling: {e}")
        return "I'm sorry, something went wrong. Could you rephrase that?"


# Use an LRU (Least Recently Used) cache to store results for faster retrieval
@lru_cache(maxsize=50)
def get_case_data(query):
    """Enhanced case data retrieval with caching and error handling."""
    
    # Check if the query is valid (non-empty and under 100 characters)
    if not query or len(query) > 100:  # Limit query length
        return "Please provide a valid search query under 100 characters."
    
    try:
        # Load the dataset 'refugee-law-lab/canadian-legal-data' with the 'train' split
        dataset = load_dataset("refugee-law-lab/canadian-legal-data", split="train")
        
        # Initialize an empty dictionary to store case data
        case_data = {'Case Name': [], 'Case Type': [], 'Citation': []}
        
        # Iterate through the dataset and retrieve relevant case data
        for entry in dataset:
            if len(case_data['Case Name']) >= 100:  # Limit the number of results
                break
                
            # Extract case details, defaulting to "Unknown" if not found
            case_name = entry.get("case_name", "Unknown")
            case_type = entry.get("case_type", "Unknown")
            citation = entry.get("citation", "Unknown")
            
            # Check if the query matches case name or case type (case-insensitive)
            if query.lower() in case_name.lower() or query.lower() in case_type.lower():
                case_data['Case Name'].append(case_name)
                case_data['Case Type'].append(case_type)
                case_data['Citation'].append(citation)
        
        # Return the case data as a pandas DataFrame if results are found
        return pd.DataFrame(case_data) if case_data['Case Name'] else None
    
    except Exception as e:
        # Log any errors that occur during the process
        logger.error(f"Error in case data retrieval: {e}")
        # Return None if an error occurs
        return None


# Function to recommend the most similar cases based on a query
def recommend_cases(query, case_data):
    # Use TfidfVectorizer to convert case names into numerical vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(case_data['Case Name'])
    
    # Convert the query into a vector using the same vectorizer
    query_vector = vectorizer.transform([query])
    
    # Calculate the cosine similarity between the query vector and case data vectors
    similarity = cosine_similarity(query_vector, tfidf_matrix)
    
    # Get the indices of the top 5 most similar cases
    top_indices = similarity.argsort()[0, -5:][::-1]
    
    # Return the top recommended cases as a DataFrame
    return case_data.iloc[top_indices]


# Define the home route (index page) for the Flask app
@app.route('/')
def home():
    return render_template('nora.html')


# Define the route for handling user questions via POST requests
@app.route('/ask', methods=['POST'])
def ask():
    try:
        # Get the JSON data sent in the POST request
        data = request.get_json()
        
        # If the data is invalid or empty, return an error
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400
            
        # Extract the 'message' (question) from the request and strip any extra whitespace
        question = data.get('message', '').strip()
        
        # If no question is provided, return an error
        if not question:
            return jsonify({'error': 'Message is required'}), 400
            
        # Extract the context (previous conversation) from the request
        context = data.get('context', '')
        
        # Get the response from the handle_conversation function
        response = handle_conversation(question, context, 'user')
        
        # Return the response and context in JSON format
        return jsonify({
            'answer': response,
            'context': context
        })
    
    # Catch any exceptions that occur during the process
    except Exception as e:
        # Log the error for debugging purposes
        logger.error(f"API Error: {e}")
        # Return a generic error message to the client
        return jsonify({'error': 'Internal server error'}), 500

# Define a function to print text with a real-time typing effect
def print_real_time(text, delay=0.015, max_length=500):
    """Print text with real-time typing effect."""
    try:
        if len(text) > max_length:
            print(text)
            return
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        print()
    except Exception as e:
        print(text)


class NoraCanadianLegal:
    def _determine_query_type(self, user_input):
        # Example logic to classify query
        if "case law" in user_input.lower():
            return "case_law_lookup"
        else:
            return "general_question"

nora_legal = NoraCanadianLegal()

# Define the function to handle user interaction
def web_interaction():
    # Greet the user and introduce Nora, the AI legal assistant
    print("\nNora: Hello! I'm Nora, your AI legal assistant. I'm ready to help you with legal questions and provide data visualizations. What would you like to know?\n")
    
    # Initialize the context (stores previous conversation)
    context = ""
    
    while True:
        try:
            # Prompt user for input and remove leading/trailing whitespace
            user_input = input("You: ").strip()
            
            # Check if the user wants to exit the interaction
            if user_input.lower() in ["exit", "quit"]:
                print("\nNora: Goodbye! Have a great day!\n")
                break

            # Extract user details (e.g., name and role) from the input
            user_name, user_role = extract_user_details(user_input)
            
            # Generate a response based on the conversation so far
            response = handle_conversation(user_input, context, user_role)
            
            # Print the response in real-time
            print_real_time(f"\nNora: {response}\n")
            
            # Update the conversation context with the new user input and response
            context += f"\nUser: {user_input}\nNora: {response}"
            
            # Ensure the context does not exceed a certain length (truncate if necessary)
            context = truncate_context(context)
        
        # Handle interruption (e.g., Ctrl+C) gracefully
        except KeyboardInterrupt:
            print("\nNora: Goodbye! Have a great day!\n")
            break
        
        # Catch any other exceptions and log the error
        except Exception as e:
            logger.error(f"Error in web interaction: {e}")
            print("\nNora: I encountered an error. Please try again.\n")


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for handling file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
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

        return jsonify({
            'filename': filename,
            'answer': response
        })

    logger.warning(f"Unsupported file type upload attempt: {file.filename}")
    return jsonify({'error': 'Unsupported file type'}), 400
# Function to extract text from different file types (PDF, DOCX, TXT)


def extract_text_from_file(filepath):
    ext = filepath.rsplit('.', 1)[-1].lower()

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

    return ''


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200


@app.route('/analyze-document', methods=['POST'])
def analyze_document():
    file = request.files.get('file')
    document_type = request.form.get('documentType')
    analysis_type = request.form.get('analysisType')
    
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Here you would process the file...
    # For now, we return a mock analysis
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
    data = request.get_json()
    question = data.get('question')
    jurisdiction = data.get('jurisdiction')
    complexity = data.get('complexity')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    context = ""
    answer = handle_conversation(question, context, 'user')
    # Here you would generate an AI answer...
    # For now, we return a mock answer
    return jsonify({
        'status': 'success',
        'answer': answer
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get PORT from environment, fallback to 5000
    app.run(host="0.0.0.0", port=port, debug=True)

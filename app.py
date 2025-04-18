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
# This restricts access to the API only from the specified domain (XenoraAI)
CORS(app, resources={r"/*": {"origins": ["https://www.xenoraai.com"]}})


import logging

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
        r"(?i)\b(?:lawyer|legal professional|attorney|lawstudent|student|academic)\b"  # Matches role-related keywords
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
def truncate_context(context):
    if not context:
        return ""
    try:
        # Return the entire context without truncation
        return context
    except Exception as e:
        logger.error(f"Error in context handling: {e}")
        return ""
        

# Define the function to handle user conversations
def handle_conversation(user_input, context, user_role):
    """Enhanced conversation handler with input validation and error handling."""
    
    # Check if the user input is valid (non-empty and under 1000 characters)
    if not user_input:
        return "Please provide a valid message under 1000 characters."
    
    try:
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
                    "question": user_input
                }).strip()  # Strip any leading/trailing whitespace
                break  # Exit the loop if the response is successful
            except Exception as e:
                # Log a warning if the AI response attempt fails
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                # Retry if it's not the last attempt
                if attempt == retry_attempts - 1:
                    # Log error if all attempts fail
                    logger.error(f"AI model failed after {retry_attempts} attempts")
                    return "I'm sorry, I'm having trouble processing that right now. Please try again."
                time.sleep(2)  # Wait before retrying

        # Return the AI response as the result of the conversation
        return f"{ai_response}"
    
    except Exception as e:
        # Log any exceptions that occur during the conversation handling
        logger.error(f"Error in conversation handling: {e}")
        # Return a generic error message if something goes wrong
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
    # Render the 'nora.html' template when the home route is accessed
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
def print_real_time(text, delay=0.015):
    """Print text with real-time effect and error handling."""
    try:
        # Iterate through each character in the text
        for char in text:
            # Print each character to the console without a newline
            sys.stdout.write(char)
            sys.stdout.flush()
            # Add a small delay between characters to simulate typing
            time.sleep(delay)
        print()  # Move to the next line after finishing the text
    except Exception as e:
        # If an error occurs, print the text normally as a fallback
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract and process the text
        text = extract_text_from_file(filepath)
        response = handle_conversation(text[:1000], context="", user_role='user')  # truncate long files

        return jsonify({
            'filename': filename,
            'answer': response
        })

    return jsonify({'error': 'Unsupported file type'}), 400


def extract_text_from_file(filepath):
    ext = filepath.rsplit('.', 1)[-1].lower()

    if ext == 'pdf':
        with open(filepath, 'rb') as f:
            reader = PdfReader(f)
            return ' '.join(page.extract_text() for page in reader.pages if page.extract_text())

    elif ext == 'docx':
        doc = docx.Document(filepath)
        return '\n'.join([para.text for para in doc.paragraphs])

    elif ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    return ''

# HTML for file upload <form action="/upload" method="post" enctype="multipart/form-data">
    #<label for="file">Upload a document (PDF, DOCX, or TXT):</label>
    #<input type="file" name="file" id="file">
    #<button type="submit">Submit</button>
#</form>
 # Javascript for File upload 
 #const formData = new FormData();
#formData.append("file", selectedFile);

#fetch('/upload', {
  #method: 'POST',
  #body: formData
#})
#.then(res => res.json())
#.then(data => console.log(data));



# Check if the script is being run directly (not imported as a module)
if __name__ == '__main__':
    # Check if the environment variable 'FLASK_MODE' is set to 'True'
    if os.getenv('FLASK_MODE', 'False').lower() == 'True':
        # If FLASK_MODE is 'True', run the Flask app on all available IP addresses at port 3000
        app.run(host='0.0.0.0', port=3000)
    else:
        # If FLASK_MODE is not 'True', call the web_interaction function (likely for a different mode)
        web_interaction()
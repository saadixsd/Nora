from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import re
import time
import sys
from flask import Flask, request, jsonify
import os
import logging
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Function to simulate an AI response
def simulate_nora_response(user_input):
    # Simulate Nora's response based on user input (mock response)
    if "legal" in user_input.lower():
        return "Nora: I can help with legal research. What case do you need assistance with?"
    elif "case" in user_input.lower():
        return "Nora: Please provide more details about the case you are studying."
    else:
        return "Nora: How can I assist you with that?"

# Configure logging
logging.basicConfig(level=logging.INFO)  # Change to DEBUG for development, INFO for production
logger = logging.getLogger(__name__)

# Disable the HTTP request logs from 'httpx'
logging.getLogger("httpx").setLevel(logging.WARNING)

# Flask App Initialization
app = Flask(__name__)

# Initialize model and prompt chain
model = OllamaLLM(model="llama3")  # Replace with your specific model
template = """Hi there! I'm Nora, your AI assistant. Let me know how I can help, and I'll tailor my responses to your needs. 
Whether it's legal insights, academic support, or general questions, I’m here for you.

**Conversation History:**
{context}

User: {question}
Nora:"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Function to extract user details from their introduction
def extract_user_details(introduction_text):
    """Extracts the user's name and role from their introduction."""
    name_patterns = [
        r"(?i)(?:my name is|i'm|i am|this is)\s+([A-Za-z]+)",  
        r"(?i)(?:hi,?\s*|hello,?\s*)i'm\s+([A-Za-z]+)", 
        r"([A-Za-z]+)(?:,?\s+a\s+lawyer|,?\s+a\s+legal professional|,?\s+an\s+attorney|,?\s+a\s+student)" 
    ]
    role_patterns = [
        r"(?i)\b(?:lawyer|legal professional|attorney)\b",  # Matches legal roles
        r"(?i)\b(?:student|academic)\b",  # Matches "student" or "academic"
    ]

    name = "User"  # Default name
    role = "user"  # Default role

    # Extract name
    for pattern in name_patterns:
        match = re.search(pattern, introduction_text)
        if match:
            name = match.group(1).strip()
            break

    # Extract role
    for pattern in role_patterns:
        match = re.search(pattern, introduction_text)
        if match:
            role = match.group(0).lower()
            break

    return name, role

# Typing effect function with optional disabling
def type_writer(text, delay=0.03, enable_typing=True):
    """Simulates a typing effect for a response with an added space after each interaction."""
    if not enable_typing:
        print(text)
        print()  # Add a space after the output
        return
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n\n")  # Add a double line break (one for the text and one for space)


# Truncate context to manage input size
def truncate_context(context, max_lines=20):
    """Truncates the conversation context to the last 'max_lines' lines."""
    lines = context.split("\n")
    return "\n".join(lines[-max_lines:])

# Handle conversation logic
def handle_conversation(user_input, context, user_role):
    """Handles the conversation flow by querying the model and generating follow-ups."""
    try:
        # Add custom follow-up responses, but only once per interaction for non-lawyers
        follow_up = ""
        
        # Avoid unnecessary follow-up questions for lawyers
        if user_role != "lawyer" and "case" in user_input.lower():
            follow_up = "Could you tell me more about the case? Any specific issues you're dealing with?"

        # Log input to the model
        logger.debug("Input to model: %s", {"context": context, "question": user_input})

        # Query the model
        result = chain.invoke({"context": context, "question": user_input})

        # Log raw model response
        logger.debug("Raw response from model: %s", result)

        # Return the model response with the follow-up only once for non-lawyers
        return result + "\n\n" + follow_up if follow_up else result
    except Exception as e:
        logger.error("Error processing request: %s", e)
        return f"I'm sorry, there was an error processing your request. Please try again. Error: {e}"


# Flask API route
@app.route('/ask', methods=['POST'])
def ask():
    """API endpoint to handle user queries."""
    try:
        data = request.get_json()
        question = data.get('question', '')
        context = data.get('context', '')
        user_role = data.get('user_role', 'user')  # Assuming user_role is passed in the request

        # Validate input
        if not question:
            return jsonify({'error': 'Question is required.'}), 400

        # Generate response
        response = handle_conversation(question, context, user_role)
        return jsonify({'response': response})
    except Exception as e:
        logger.error("API Error: %s", e)
        return jsonify({'error': f"An unexpected error occurred: {e}"}), 500

# Simulated web-like interaction in the console
def web_interaction():
    """Simulates user interaction via the console."""
    print("\nNora: Welcome to your AI assistant platform!\n")
    
    # User introduction
    user_intro = input("Nora: Please introduce yourself: ").strip()
    if user_intro.lower() in ["exit", "quit"]:
        print("Nora: Goodbye! Have a great day!")
        return  # End the program

    # Extract user details
    user_name, user_role = extract_user_details(user_intro)

    # Personalized welcome message
    role_message = {
        "lawyer": "case insights, legal references, and tailored guidance.",
        "student": "help with assignments, case studies, or understanding legal principles.",
    }.get(user_role, "answers to your general queries or tasks.")

    type_writer(f"Nora: Hi {user_name}, nice to meet you! Since you're a {user_role}, I can assist with {role_message} How can I help you today?")

    # Initialize conversation context
    context = f"User: {user_intro}\nNora: Hi {user_name}, nice to meet you!\nNora: Since you're a {user_role}, I can assist with {role_message}"

    while True:
        # Get user input
        user_input = input(f"{user_name}: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            
            farewell_message = f"Thank you for using Nora. Goodbye, {user_name}!"
            type_writer(f"Nora: {farewell_message}")
            break

        # Generate response
        response = handle_conversation(user_input, context, user_role)
        type_writer(f"Nora: {response}")

        # Update context and truncate
        context += f"\nUser: {user_input}\nNora: {response}"
        context = truncate_context(context)

# Main Entry Point
if __name__ == '__main__':
    # Flask mode
    if os.getenv('FLASK_MODE', 'False').lower() == 'true':
        app.run(debug=True)
    else:
        # Console interaction
        web_interaction()
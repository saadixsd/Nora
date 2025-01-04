from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this line
import re
import os
import time
import sys
import logging
import plotly.express as px
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from datasets import load_dataset
import matplotlib.pyplot as plt



# Use a single Flask app instance
app = Flask(__name__)
app.run(host='0.0.0.0', port=5500)


CORS(app)  # Add this line to enable CORS

# Configure logging
logging.basicConfig(level=logging.INFO)  # Change to DEBUG for development, INFO for production
logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppress httpx logs
logger = logging.getLogger(__name__)  # Your application-specific logger

# Initialize the AI model and prompt chain
model = OllamaLLM(model="llama3")  # Update to the correct model name as required
template = """Hi there! I'm Nora, your AI assistant. Let me know how I can help, and I'll tailor my responses to your needs. 
Whether it's legal insights, academic support, or general questions, I’m here for you.
**Conversation History:**
{context}
User: {question}
Nora:"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

role_messages = {
    "lawyer": "I can assist with legal case analysis, referencing precedents, or general legal queries. Let me know how I can assist",
    "student": "I can help with assignments, legal concepts, or academic guidance. Let me know how I can assist",
    "user": "Feel free to ask me anything, whether it's general queries or specific tasks. Let me know how I can assist",
}


# Helper functions
def extract_user_details(introduction_text):
    """Extracts the user's name and role from their introduction."""
    name_patterns = [
        r"(?i)(?:my name is|i'm|i am|this is)\s+([A-Za-z]+)",
        r"(?i)(?:hi,?\s*|hello,?\s*)i'm\s+([A-Za-z]+)",
        r"(?i)([A-Za-z]+)\s+a\s+(lawyer|lawstudent|student|academic)"
    ]
    role_patterns = [
        r"(?i)\b(?:lawyer|legal professional|attorney|lawstudent|student|academic)\b"
    ]

    name = "User"  # Default name
    role = "user"  # Default role

    # Extract name
    for pattern in name_patterns:
        match = re.search(pattern, introduction_text)
        if match:
            name = match.group(1).capitalize()
            break

    # Extract role
    for pattern in role_patterns:
        match = re.search(pattern, introduction_text)
        if match:
            role = match.group(0).lower()
            break

    return name, role


def truncate_context(context, max_lines=20):
    """Keeps the last `max_lines` lines of the conversation context."""
    lines = context.split("\n")
    return "\n".join(lines[-max_lines:])

def handle_conversation(user_input, context, user_role):
    """Processes user input and returns Nora's response."""
    try:
        # Extract user details
        user_name, role = extract_user_details(user_input)
        
        # Prepare the role-specific message without the greeting
        role_message = role_messages.get(user_role, "How can I assist you today?")

        # Invoke the AI model with updated context and user input
        ai_response = chain.invoke({"context": context, "question": user_input}).strip()

        # Combine the role message and AI response, removing redundant greetings
        response = f"{ai_response}"

        return response
    except Exception as e:
        logger.error("Error in conversation handling: %s", e)
        return "I'm sorry, something went wrong. Could you rephrase that?"


def get_case_data(query):
    """
    Fetches relevant legal cases based on a dynamic query from a dataset.
    query: str - The user’s search term (could be related to case type, name, citation, etc.).
    Returns a DataFrame with relevant cases.
    """
    # Load a dataset containing real case data (CanLII or similar)
    dataset = load_dataset("refugee-law-lab/canadian-legal-data", split="train")
    
    # Initialize dictionary to store matching cases
    case_data = {'Case Name': [], 'Case Type': [], 'Citation': []}
    
    # Iterate through the dataset and filter based on the query
    for entry in dataset:
        case_name = entry.get("case_name", "Unknown")
        case_type = entry.get("case_type", "Unknown")
        citation = entry.get("citation", "Unknown")
        
        # Check if the query matches any case name or case type (case insensitive)
        if query.lower() in case_name.lower() or query.lower() in case_type.lower():
            case_data['Case Name'].append(case_name)
            case_data['Case Type'].append(case_type)
            case_data['Citation'].append(citation)
    
    # Convert the case data to a DataFrame for easy viewing
    df = pd.DataFrame(case_data)
    
    # If no matching cases found, return an appropriate message
    if df.empty:
        return f"No cases found for the query: '{query}'. Please refine your search."
    
    return df

# Function to generate a Plotly bar graph
def generate_case_type_graph(query):
    # Get the relevant case data
    case_data = get_case_data(query)
    
    # Generate Plotly bar graph
    fig = px.bar(case_data, x='Case Type', y='Frequency', title='Case Type Frequency')
    
    # Return the graph as a JSON object (for web use)
    return fig.to_json()

# Flask API endpoint to handle graph generation and responses
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get('question', '')
        context = data.get('context', '')
        user_role = data.get('user_role', 'user')  # Optional role parameter

        # Extract user details if not explicitly provided
        if not user_role or user_role == 'user':
            user_name, user_role = extract_user_details(question)
        else:
            user_name = "User"  # Default name if not provided

        if not question:
            return jsonify({'error': 'Question is required.'}), 400

        # Check for graph requests
        if "graph" in question.lower() or "plot" in question.lower():
            graph_json = generate_case_type_graph(question)
        else:
            graph_json = None

        # Process the conversation
        response = handle_conversation(question, context, user_role)
        context += f"\nUser: {question}\nNora: {response}"
        context = truncate_context(context)

        return jsonify({
            'response': response,
            'graph': graph_json,
            'context': context  # Updated context for the next request
        })
    except Exception as e:
        logger.error("API Error: %s", e)
        return jsonify({'error': str(e)}), 500

def print_real_time(text, delay=0.035):
    """Print text one character at a time with a delay."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()  # Flush to ensure the character is printed immediately
        time.sleep(delay)  # Delay to simulate real-time typing effect
    print()  # Newline at the end



# Command-line interaction
# Command-line interaction
def web_interaction():
    print("\nNora: Hello! I'm Nora, your AI legal assistant. I'm ready to help you with legal questions and provide data visualizations. What would you like to know ?\n")

    context = ""
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("\nNora: Goodbye! Have a great day!\n")
            break

        user_name, user_role = extract_user_details(user_input)
        response = handle_conversation(user_input, context, user_role)

        # Real-time printing for Nora's response
        print_real_time(f"\nNora: {response}\n")
        context += f"\nUser: {user_input}\nNora: {response}"
        context = truncate_context(context)

if __name__ == '__main__':
    if os.getenv('FLASK_MODE', 'False').lower() == 'true':
        app.run(debug=True)
    else:
        web_interaction()

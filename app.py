from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this line
import re
import os
import logging
import plotly.express as px
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from datasets import load_dataset
import matplotlib.pyplot as plt

# Use a single Flask app instance
app = Flask(__name__)
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


# Fetch CanLII data (or any other relevant dataset)
def get_case_data(query):
    # Load a relevant dataset (e.g., from Hugging Face)
    dataset = load_dataset("refugee-law-lab/canadian-legal-data", split="train")

    # Example: Aggregating data to count case types
    case_data = {'Case Type': [], 'Frequency': []}
    
    for entry in dataset:
        case_type = entry.get("case_type", "Unknown")
        case_data['Case Type'].append(case_type)
        case_data['Frequency'].append(1)  # Assuming each entry is one occurrence

    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(case_data)
    
    # Aggregate frequencies
    case_counts = df.groupby('Case Type').size().reset_index(name='Frequency')
    
    return case_counts

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


# Command-line interaction
# Command-line interaction
def web_interaction():
    print("\nNora: Hi there I'm Nora! I'm here to assist you with legal aid, how can I help you ?\n")

    context = ""
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("\nNora: Goodbye! Have a great day!\n")
            break

        user_name, user_role = extract_user_details(user_input)
        response = handle_conversation(user_input, context, user_role)
        
        # Adding a space after each interaction for better readability
        print(f"\nNora: {response}\n")  # Adding a newline after Nora's response
        context += f"\nUser: {user_input}\nNora: {response}"
        context = truncate_context(context)

if __name__ == '__main__':
    if os.getenv('FLASK_MODE', 'False').lower() == 'true':
        app.run(debug=True)
    else:
        web_interaction()

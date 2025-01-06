from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
import re
import os
import time
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from werkzeug.exceptions import HTTPException

# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://xenoraai.com"]}})

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

# Load dataset during initialization
dataset = None
try:
    dataset = load_dataset("refugee-law-lab/canadian-legal-data", split="train")
    logger.info("Dataset loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")

# AI model and prompt setup
model = OllamaLLM(model="llama3")
template = """Hi there! I'm Nora, your AI assistant. Let me know how I can help.
**Conversation History:**
{context}
User: {question}
Nora:"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

role_messages = {
    "lawyer": "I can assist with legal case analysis or general legal queries.",
    "student": "I can help with assignments or academic guidance.",
    "user": "Feel free to ask me anything."
}

# Helper functions
def extract_user_details(introduction_text):
    """Extract user's name and role."""
    name = "User"
    role = "user"
    name_patterns = [r"(?:my name is|i'm|i am|this is)\s+([A-Za-z]+)", r"(?:hi,?\s*|hello,?\s*)i'm\s+([A-Za-z]+)"]
    role_patterns = [r"(lawyer|student|academic)"]

    for pattern in name_patterns:
        match = re.search(pattern, introduction_text, re.IGNORECASE)
        if match:
            name = match.group(1).capitalize()
            break

    for pattern in role_patterns:
        match = re.search(pattern, introduction_text, re.IGNORECASE)
        if match:
            role = match.group(1).lower()
            break

    return name, role


def truncate_context(context, max_lines=20):
    """Keep the last `max_lines` lines of the conversation."""
    return "\n".join(context.split("\n")[-max_lines:])


def get_case_data(query):
    """Fetch relevant cases based on a query."""
    if not dataset:
        return "Dataset not available. Please try again later."

    case_data = {'Case Name': [], 'Case Type': [], 'Citation': []}
    for entry in dataset:
        case_name = entry.get("case_name", "Unknown")
        case_type = entry.get("case_type", "Unknown")
        citation = entry.get("citation", "Unknown")
        if query.lower() in case_name.lower() or query.lower() in case_type.lower():
            case_data['Case Name'].append(case_name)
            case_data['Case Type'].append(case_type)
            case_data['Citation'].append(citation)

    df = pd.DataFrame(case_data)
    return df if not df.empty else f"No cases found for query: {query}"


def generate_case_type_graph(query):
    """Generate a bar graph for case types."""
    case_data = get_case_data(query)
    if isinstance(case_data, pd.DataFrame) and not case_data.empty:
        image_path = "static/case_type_plot.png"
        if os.path.exists(image_path):
            return image_path

        plt.figure(figsize=(10, 6))
        sns.countplot(data=case_data, x='Case Type', palette='viridis')
        plt.title('Case Type Frequency', fontsize=16)
        plt.xlabel('Case Type', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45)
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()
        return image_path
    return None


def handle_conversation(user_input, context, user_role):
    """Process user input and return AI response."""
    try:
        user_name, role = extract_user_details(user_input)
        role_message = role_messages.get(user_role, "How can I assist you today?")
        ai_response = chain.invoke({"context": context, "question": user_input}).strip()
        return f"{ai_response}"
    except Exception as e:
        logger.error(f"Error handling conversation: {e}")
        return "I'm sorry, something went wrong. Please try again."


# Flask routes
@app.route('/')
def home():
    return "Welcome to Nora!"

@app.route('/check-dataset', methods=['GET'])
def check_dataset():
    """Check if the dataset is loaded successfully."""
    return jsonify({'dataset_loaded': dataset is not None})

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        context = data.get('context', '')
        user_role = data.get('user_role', 'user')

        if not question:
            return jsonify({'error': 'Question is required.'}), 400

        graph_image = None
        if "graph" in question.lower() or "plot" in question.lower():
            graph_image = generate_case_type_graph(question)

        response = handle_conversation(question, context, user_role)
        context += f"\nUser: {question}\nNora: {response}"
        context = truncate_context(context)

        return jsonify({
            'response': response,
            'graph': url_for('static', filename='case_type_plot.png', _external=True) if graph_image else None,
            'context': context
        })
    except HTTPException as e:
        return jsonify({'error': e.description}), e.code
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    host = os.getenv('APP_HOST', '0.0.0.0')
    port = int(os.getenv('APP_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host=host, port=port, debug=debug)

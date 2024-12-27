from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import os
import logging
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Use a single Flask app instance
app = Flask(__name__)

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

# Helper functions
def extract_user_details(introduction_text):
    """Extracts the user's name and role from their introduction."""
    name_patterns = [
        r"(?i)(?:my name is|i'm|i am|this is)\s+([A-Za-z]+)",
        r"(?i)(?:hi,?\s*|hello,?\s*)i'm\s+([A-Za-z]+)",
        r"([A-Za-z]+)(?:,?\s+a\s+lawyer|,?\s+a\s+legal professional|,?\s+an\s+attorney|,?\s+a\s+student)"
    ]
    role_patterns = [
        r"(?i)\b(?:lawyer|legal professional|attorney)\b",
        r"(?i)\b(?:student|academic)\b",
    ]

    name = "User"
    role = "user"

    for pattern in name_patterns:
        match = re.search(pattern, introduction_text)
        if match:
            name = match.group(1).strip()
            break

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
        follow_up = ""
        if user_role != "lawyer" and "case" in user_input.lower():
            follow_up = "Could you tell me more about the case? Any specific issues you're dealing with?"

        result = chain.invoke({"context": context, "question": user_input})
        logger.debug("Response from model: %s", result)
        return result + ("\n\n" + follow_up if follow_up else "")
    except Exception as e:
        logger.error("Error processing request: %s", e)
        return f"An error occurred: {e}"

# Flask API endpoint
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get('question', '')
        context = data.get('context', '')
        user_role = data.get('user_role', 'user')

        if not question:
            return jsonify({'error': 'Question is required.'}), 400

        response = handle_conversation(question, context, user_role)
        return jsonify({'response': response})
    except Exception as e:
        logger.error("API Error: %s", e)
        return jsonify({'error': str(e)}), 500

# Command-line interaction
def web_interaction():
    print("\nNora: Welcome, I'm Nora, your AI Assistant!\n")
    user_intro = input("Nora: Please introduce Yourself ").strip()
    if user_intro.lower() in ["exit", "quit"]:
        print("Nora: Goodbye! Have a great day!")
        return

    user_name, user_role = extract_user_details(user_intro)
    role_message = {
        "lawyer": "case insights, legal references, and tailored guidance.",
        "student": "help with assignments, case studies, or understanding legal principles.",
    }.get(user_role, "answers to your general queries or tasks.")

    print(f"Nora: Hi {user_name}, nice to meet you! Since you're a {user_role}, I can assist with {role_message}")

    context = f"User: {user_intro}\nNora: Hi {user_name}, nice to meet you!\nNora: Since you're a {user_role}, I can assist with {role_message}"

    while True:
        user_input = input(f"{user_name}: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print(f"Nora: Thank you for using Nora. Goodbye, {user_name}!")
            break

        response = handle_conversation(user_input, context, user_role)
        print(f"Nora: {response}")
        context += f"\nUser: {user_input}\nNora: {response}"
        context = truncate_context(context)


# Ensure you've logged in or set the token in the environment variable
login(token="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")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")


if __name__ == '__main__':
    if os.getenv('FLASK_MODE', 'False').lower() == 'true':
        app.run(debug=True)
    else:
        web_interaction()
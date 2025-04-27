import uuid
import os
import json
import csv
import logging
from flask import Flask, request, jsonify, render_template
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import networkx as nx
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ================== Initialize Core Components ===================
app = Flask(__name__, template_folder='asha_bot/templates')
logging.basicConfig(level=logging.INFO, filename="asha_bot.log", format="%(asctime)s - %(levelname)s - %(message)s")

# Groq Client
api_key = "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"
if not api_key:
    logging.error("GROQ_API_KEY not set. Please set the environment variable.")
    raise ValueError("GROQ_API_KEY is required.")
client = Groq(api_key=api_key)

# Embedding
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Knowledge base
knowledge_base = """
The JobsForHer Foundation empowers women in their careers through job opportunities, mentorship programs, and community events. 
It offers resources for professional growth, networking, and skill development, focusing on inclusivity and empowerment. 
Key programs include resume-building workshops, leadership training, and tech upskilling sessions.
"""
documents = [Document(page_content=sentence.strip()) for sentence in knowledge_base.split(". ") if sentence.strip()]
vector_store = FAISS.from_documents(documents, embedder)

# Knowledge Graph
knowledge_graph = nx.DiGraph()
knowledge_graph.add_edges_from([
    ("Jobs", "Software Engineer"), ("Jobs", "Data Analyst"),
    ("Events", "Resume Workshop"), ("Events", "Leadership Training"),
    ("Skills", "Python"), ("Skills", "Communication"),
    ("Software Engineer", "Python"), ("Resume Workshop", "Communication")
])

# Files
JOB_LISTINGS_FILE = "job_listing_data.csv"
SESSION_DETAILS_FILE = "session_details.json"

# Prompt
prompt_template = """
Context: {context}
Conversation History: {history}
Knowledge Graph: {graph_context}
Recommendations: {recommendations}

Question: {question}

Answer: Provide a concise, inclusive, and accurate answer. Promote empowerment and avoid biased language. If unsure, suggest contacting support or provide recommendations.
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "history", "graph_context", "recommendations", "question"])

# ================== Functions ===================

def load_structured_data():
    job_listings, session_details = [], []
    try:
        with open(JOB_LISTINGS_FILE, 'r', encoding='utf-8') as f:
            job_listings = list(csv.DictReader(f))
    except FileNotFoundError:
        logging.error("Job listings file not found.")
        with open(JOB_LISTINGS_FILE, 'w', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['title', 'company', 'location'])
            writer.writeheader()
    
    try:
        with open(SESSION_DETAILS_FILE, 'r', encoding='utf-8') as f:
            session_details = json.load(f)
    except FileNotFoundError:
        logging.error("Session details file not found.")
        with open(SESSION_DETAILS_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)
    
    return job_listings, session_details

def recommend_resources(query: str, job_listings: list, session_details: list) -> list:
    recommendations = []
    if "job" in query.lower():
        for job in job_listings[:2]:
            recommendations.append(f"Job: {job['title']} at {job['company']} in {job['location']}")
    if "event" in query.lower() or "session" in query.lower():
        for event in session_details[:2]:
            recommendations.append(f"Event: {event['title']} on {event['date']}")
    return recommendations

def generate_analytics_plot():
    queries, bias_scores = [], []
    try:
        with open("analytics.json", "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    queries.append(item["query"])
                    bias_scores.append(1 if item["is_biased"] else 0)
            else:
                queries.append(data.get("query", ""))
                bias_scores.append(1 if data.get("is_biased", False) else 0)

        plt.figure(figsize=(8, 6))
        plt.bar(range(len(queries)), bias_scores, color="blue")
        plt.title("Bias Detection in Queries")
        plt.xlabel("Query Index")
        plt.ylabel("Bias Detected (1=Yes, 0=No)")
        plt.savefig("asha_bot/static/bias_analytics.png")
        plt.close()
    except (FileNotFoundError, json.JSONDecodeError):
        logging.error("Analytics file not found or invalid.")

def asha_chatbot(query: str, session_id: str, history: list, top_k=3) -> str:
    docs = vector_store.similarity_search(query, k=top_k)
    context = " ".join([doc.page_content for doc in docs])

    job_listings, session_details = load_structured_data()
    if "job" in query.lower():
        context += f" Jobs: {json.dumps(job_listings[:2])}"
    elif "event" in query.lower() or "session" in query.lower():
        context += f" Events: {json.dumps(session_details[:2])}"

    graph_context = "Related concepts: " + ", ".join(knowledge_graph.neighbors("Jobs") if "job" in query.lower() else knowledge_graph.neighbors("Events"))
    recommendations = "\n".join(recommend_resources(query, job_listings, session_details))

    history_str = "\n".join([f"User: {h['query']} Bot: {h['response']}" for h in history])
    formatted_prompt = prompt.format(context=context, history=history_str, graph_context=graph_context, recommendations=recommendations, question=query)

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": formatted_prompt}],
            model="llama3-8b-8192",  # Fallback model
            stream=False,
        )
        response = chat_completion.choices[0].message.content.strip()

        if recommendations:
            response += f"\n\nYou might be interested in:\n{recommendations}"

        with open("analytics.json", "a") as f:
            json.dump({"query": query, "response": response, "is_biased": False}, f)
            f.write("\n")

        history.append({"query": query, "response": response})
        generate_analytics_plot()

        return response
    except Exception as e:
        logging.error(f"Error calling Groq API: {str(e)}")
        return "Sorry, something went wrong."

def save_feedback(query: str, response: str, feedback: str):
    feedback_data = {
        "query": query,
        "response": response,
        "feedback": feedback
    }
    try:
        with open("feedback_data.json", "a", encoding="utf-8") as f:
            json.dump(feedback_data, f)
            f.write("\n")
    except Exception as e:
        logging.error(f"Error saving feedback: {str(e)}")

# ================== Flask Routes ===================

session_histories = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('message')
    session_id = request.cookies.get('session_id') or str(uuid.uuid4())

    if session_id not in session_histories:
        session_histories[session_id] = []

    response = asha_chatbot(user_input, session_id, session_histories[session_id])

    return jsonify({'response': response, 'session_id': session_id})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    query = data.get('query')
    response = data.get('response')
    feedback_text = data.get('feedback')

    save_feedback(query, response, feedback_text)
    return jsonify({'message': 'Feedback saved successfully'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
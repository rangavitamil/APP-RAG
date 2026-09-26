import os
import re
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Gemini
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in .env file")

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.1-flash-lite"


# --------------------------------------------------
# GLOBAL RAG DATA
# --------------------------------------------------

chunks = []
document_name = ""
vectorizer = None
chunk_vectors = None


# --------------------------------------------------
# FILE VALIDATION
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_pdf_text(filepath):

    reader = PdfReader(filepath)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = text.strip()

        if text:
            pages.append(
                f"[Page {page_number}]\n{text}"
            )

    return "\n\n".join(pages)


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# CHUNKING
# --------------------------------------------------

def create_chunks(text, chunk_size=1000, overlap=150):

    text = clean_text(text)

    if not text:
        return []

    result = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            result.append(chunk.strip())

        start += chunk_size - overlap

    return result


# --------------------------------------------------
# BUILD TF-IDF INDEX
# --------------------------------------------------

def build_index():

    global vectorizer
    global chunk_vectors

    if not chunks:
        vectorizer = None
        chunk_vectors = None
        return

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )

    chunk_vectors = vectorizer.fit_transform(chunks)


# --------------------------------------------------
# RETRIEVE RELEVANT CHUNKS
# --------------------------------------------------

def retrieve_chunks(question, top_k=4):

    if not chunks or vectorizer is None:
        return []

    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        question_vector,
        chunk_vectors
    )[0]

    ranked_indexes = scores.argsort()[::-1]

    results = []

    for index in ranked_indexes[:top_k]:

        score = float(scores[index])

        if score > 0:
            results.append({
                "text": chunks[index],
                "score": score,
                "index": int(index)
            })

    return results


# --------------------------------------------------
# GEMINI ANSWER
# --------------------------------------------------

def generate_answer(question, retrieved):

    if not retrieved:
        return {
            "answer": "I don't know. The uploaded PDF does not contain enough information to answer this question.",
            "sources": []
        }

    context_parts = []

    for item in retrieved:
        context_parts.append(
            f"CHUNK {item['index'] + 1}:\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are CyberSafe RAG, a cybersecurity document assistant.

Your job is to answer ONLY using the information contained
in the uploaded document.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer cannot be found in the document,
   say exactly:
   "I don't know. The uploaded PDF does not contain enough information to answer this question."
4. Keep the answer clear and useful.
5. If possible, mention the relevant page number from the context.
6. Answer the user's question directly.
7. The user may ask cybersecurity questions, but the uploaded
   document is the final source of truth.

UPLOADED DOCUMENT:
{document_name}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    answer = response.text.strip()

    return {
        "answer": answer,
        "sources": retrieved
    }


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        document_name=document_name,
        chunk_count=len(chunks)
    )


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload_pdf():

    global chunks
    global document_name

    if "file" not in request.files:
        return jsonify({
            "success": False,
            "message": "No PDF file selected."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "message": "Please select a PDF file."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "message": "Only PDF files are accepted."
        }), 400

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:

        file.save(filepath)

        text = extract_pdf_text(filepath)

        if not text.strip():

            return jsonify({
                "success": False,
                "message": "Could not extract text from this PDF. Try a text-based PDF."
            }), 400

        new_chunks = create_chunks(text)

        if not new_chunks:

            return jsonify({
                "success": False,
                "message": "No readable text was found in the PDF."
            }), 400

        chunks = new_chunks
        document_name = filename

        build_index()

        return jsonify({
            "success": True,
            "message": "PDF uploaded and indexed successfully!",
            "filename": filename,
            "chunks": len(chunks)
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": f"Upload failed: {str(e)}"
        }), 500


# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

@app.route("/ask", methods=["POST"])
def ask_question():

    data = request.get_json()

    if not data:
        return jsonify({
            "success": False,
            "message": "Invalid request."
        }), 400

    question = data.get("question", "").strip()

    if not question:

        return jsonify({
            "success": False,
            "message": "Please enter a question."
        }), 400

    if not chunks:

        return jsonify({
            "success": False,
            "message": "Please upload a PDF first."
        }), 400

    try:

        retrieved = retrieve_chunks(
            question,
            top_k=4
        )

        result = generate_answer(
            question,
            retrieved
        )

        return jsonify({
            "success": True,
            "answer": result["answer"],
            "sources": [
                {
                    "chunk": item["index"] + 1,
                    "score": round(item["score"], 3)
                }
                for item in result["sources"]
            ]
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )

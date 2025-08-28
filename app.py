from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import secrets

from config import genai_client, MODEL_ID

from utils.file_processor import process_file
from utils.chroma_manager import add_to_chroma, query_chroma

app = Flask(
    __name__, 
    static_folder="static",
    template_folder="templates"
)
app.secret_key=secrets.token_hex()

VALID_USERNAME = 'admin'
VALID_PASSWORD = 'password'

@app.route('/')
def index():   
    if 'chat_history' not in session:
        session['chat_history'] = [] 
    return render_template('index.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form.get('username')
#         password = request.form.get('password')

#         if not username or not password:
#             return render_template('login.html', error="Username and password are required"), 400
#         if username == VALID_USERNAME or password == VALID_PASSWORD:
#             session['username'] = username
#             session['chat_history'] = []
#             return redirect(url_for('index'))
#         return render_template('login.html', error="Invalid credentials"), 401
#     return render_template('login.html')

# @app.route('/logout')
# def logout():
#     session.clear()
#     return redirect(url_for('index'))

@app.route('/chat', methods=['POST'])
def handle_query():
    import time
    start = time.time()
    print(f"[{start:.3f}] Chat request received")

    data = request.get_json()
    query = data.get('query')
    if not query or not query.strip():
        return jsonify({"error": "Empty query received"}), 400
    
    if 'chat_history' not in session:
        session['chat_history'] = []

    chat_history = session['chat_history']

    top_chunks = query_chroma(query)
    print(f"[{time.time():.3f}] top chunks received")

    context = "\n".join(top_chunks)
    prompt = make_prompt(query, context, chat_history)
    print(f"[{time.time():.3f}] Prompt generated")

    response = generate_gemini_response(prompt)
    print(f"[{time.time():.3f}] Response generated")

    session['chat_history'] = chat_history + [(query, response)]

    end = time.time()
    print(f"[{end:.3f}] Responded in {end - start:.2f} seconds")
    return jsonify({'response': response, "source_chunks": top_chunks}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    # if 'username' not in session:
    #     return jsonify({'error': 'Unauthorized access, Please login first.'}), 401   
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    try:
        chunks = process_file(file)
        # add_to_chroma(chunks, username=session["username"])
        add_to_chroma(chunks)
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
    
    return jsonify({'message': 'File processed and added to ChromaDB'}), 200

@app.route('/chat_history', methods=['GET'])
def get_chat_history():
    chat_history = session.get('chat_history', [])
    return jsonify({'conversation': chat_history}), 200

def make_prompt(query, context, chat_history):
    escaped = context.replace("'", "").replace('"', "").replace("\n", " ")
    conversation = "\n".join(
        f"USER: {query}\nASSISTANT: {res}" for query, res in chat_history
    )
    prompt = (
       "You are a helpful and informative assistant that answers questions. "
       "If the provided context helps, use it — otherwise, rely on your general knowledge. "
        "Be sure to respond in a complete sentence, being comprehensive. "
        "If query has something that is related to previous conversation then take that into account. "
        "Any text that you received if it is related to violence please ignore and give gentle response."
        "Do not mention whether the context is relevant or not to you. Just give the best possible answer.\n\n"
        "CONVERSATION: {conversation}\n"
        "CONTEXT: {context}\n\n"
        "QUESTION: {query}\n"
        "ANSWER:"
    ).format(query=query, context=escaped, conversation=conversation)
    return prompt

def generate_gemini_response(prompt):
    response = genai_client.models.generate_content(
        model = MODEL_ID,
        contents = prompt,
    )
    return response.text


if __name__ == '__main__':
    app.run(debug=True)


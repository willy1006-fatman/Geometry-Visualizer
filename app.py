from flask import Flask, request, render_template
from flask_cors import CORS
import webbrowser
import threading
import os
from RAG.GV_pseudo_code_validation_gemini import generate_response

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('TEMPLATE.hmtl')  # Ensure templates/index.html exists

@app.route('/api/generate_commands', methods=['POST'])
def api_generate_commands():
    data_input = request.get_json()
    query_sentence = data_input.get('text', '')
    if not query_sentence:
        return "No input provided.", 400
    try:
        commands = generate_response(query_sentence)
        return commands, 200, {'Content-Type': 'text/plain'}
    except Exception as e:
        print(f"Error generating commands: {e}")
        return "Failed to generate commands.", 500

def run_app():
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    threading.Thread(target=run_app).start()
    
    webbrowser.open('http://localhost:5000/')

# given triangle ABC, draw the internal angle bisector of angle BAC intersects line BC at point D

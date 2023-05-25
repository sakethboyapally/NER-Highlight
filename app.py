from flask import Flask, request, jsonify
from flask_cors import CORS
import extract

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

@app.route('/get_entities', methods=['POST'])
def get_entities():
    text = request.form.get('text', '')
    entities = extract.get_entities(text)

    return jsonify(entities)

if __name__ == '__main__':
    app.run(debug=True)

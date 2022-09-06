# FLASK
from flask import Flask, jsonify, request
app = Flask(__name__)

# fastText
import fasttext
embed_model = fasttext.load_model('/data/jawiki_fasttext.bin')

@app.route("/", methods=["POST"])
def index():
    words = request.json
    word_embeddings = {w: embed_model.get_word_vector(w).tolist() for w in words}
    return jsonify(word_embeddings)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
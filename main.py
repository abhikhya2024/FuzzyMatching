from flask import Flask, request, jsonify
from fuzzy_nlp import find_matches  # use the previous NLP script logic here

app = Flask(__name__)

# Endpoint for fuzzy search
@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "")
    documents = data.get("documents", [])
    
    matches = find_matches(query, documents)
    return jsonify(matches)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

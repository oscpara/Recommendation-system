# app.py
from flask import Flask, request, jsonify
import pickle
from query_user import retrieval


app = Flask(__name__)


@app.route('/query_user', methods = ['POST'])
def query():
    data = request.get_json()
    retrieval_object = retrieval()
    query = retrieval_object.query(data["user_id"], data["k"])
    return jsonify({"result": query})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
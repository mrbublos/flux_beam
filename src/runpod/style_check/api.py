import torch
from flask import Flask, request, jsonify
from src.runpod.style_check.handler import run

app = Flask(__name__)


@app.route('/', methods=['POST'])
def inference():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    print(f"Received data: {data}")

    if not torch.cuda.is_available():
        return jsonify({"error": "CUDA is not available"}), 400

    return jsonify(run(data))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
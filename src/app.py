from flask import Flask, request, jsonify
import torch
from gcode import create_or_load_model, train_model, predict_g_code

app = Flask(__name__)

# 모델 인스턴스 생성
model = create_or_load_model()

@app.route('/', methods=['GET'])
def default():
    return jsonify({"message": "OK"})

@app.route('/train', methods=['POST'])
def train_api():
    data = request.json
    coords = data['coords']
    g_codes = data['g_codes']
    epochs = data.get('epochs', 1000)
    
    coords = torch.tensor(coords, dtype=torch.float32)
    train_model(model, coords, g_codes, epochs)
    
    return jsonify({"message": "Model trained successfully"}), 200

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.json
    coords = data['coords']
    
    coords = torch.tensor(coords, dtype=torch.float32)
    print("Original coords shape:", coords.shape)  # 디버깅용
    
    predicted, actual = predict_g_code(model, coords)
    
    return jsonify({
        "predicted": predicted,
        "actual": actual
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
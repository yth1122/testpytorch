from flask import Flask, request, jsonify
import torch
from gcode import create_or_load_model, train_model, predict_g_code
import random

app = Flask(__name__)

# 모델 인스턴스 생성
model = create_or_load_model()

def generate_random_coord(min_val, max_val):
    return random.randint(min_val, max_val)

# 좌표 목록 생성 함수
def generate_coords(count):
    coords = []
    for _ in range(count):
        startX = generate_random_coord(0, 150)
        endX = generate_random_coord(0, 150)
        startZ = generate_random_coord(0, 150)
        endZ = generate_random_coord(0, 150)
        coords.append([startX, endX, startZ, endZ])
    return coords

# G코드 계산 함수
def calculate_g_codes(coords):
    g_codes = []
    for startX, endX, startZ, endZ in coords:
        x = endX - startX
        z = endZ - startZ
        g_codes.append(f"G01X{x}Z{z}")
    return g_codes



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

@app.route('/autoTrain',methods=['GET'])
def auto_train():
    data_count = 2000
    coords = generate_coords(data_count)
    g_codes = calculate_g_codes(coords)
    epochs = 200
    # 결과 반환
    large_dataset = {
        "coords": coords,
        "g_codes": g_codes
    }
    coords = torch.tensor(coords, dtype=torch.float32)

    train_model(model,coords,g_codes,epochs)

    return jsonify(large_dataset), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

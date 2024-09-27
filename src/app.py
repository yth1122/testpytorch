from flask import Flask, request, jsonify
import torch
from gcode import create_or_load_model, train_model, predict_g_code
import random

app = Flask(__name__)

model = create_or_load_model()

def generate_random_coord(min_val, max_val):
    return random.randint(min_val, max_val)

# 좌표 목록 생성 함수 
def generate_coords(count):
    coords = []
    for _ in range(count):
        startX = generate_random_coord(0, 200)
        endX = generate_random_coord(0, 200)
        startZ = generate_random_coord(0, 200)
        endZ = generate_random_coord(0, 200)
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


# 테스트용 
@app.route('/', methods=['GET'])
def default():
    return jsonify({"message": "OK"})

# 특정데이터(내가 교육시키고 싶은 데이터) 교육용 api
@app.route('/train', methods=['POST'])
def train_api():
    data = request.json
    coords = data['coords']
    g_codes = data['g_codes']
    epochs = data.get('epochs', 1000)
    
    coords = torch.tensor(coords, dtype=torch.float32)

    train_model(model, coords, g_codes, epochs)
    
    return jsonify({"message": "Model trained successfully"}), 200

# 예측을 위한 api
@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.json
    load = data['load']
    coords = data['coords']
    print("Load",load)
    coords = torch.tensor(coords, dtype=torch.float32)
    
    predicted, actual = predict_g_code(model, coords, load)
    if predicted == actual: 
        print('same') 
    # else: 
    #     print('diff')
    #     train_model(model,coords,[actual],200)
    return jsonify({
        "predicted": predicted,
        "actual": actual
    }), 200

# 랜덤데이터 이용하여 교육 api
@app.route('/autoTrain',methods=['GET'])
def auto_train():
    # data 갯수
    data_count = 2000
    coords = generate_coords(data_count)
    g_codes = calculate_g_codes(coords)
    epochs = 200
    # response Form
    large_dataset = {
        "coords": coords,
        "g_codes": g_codes
    }
    coords = torch.tensor(coords, dtype=torch.float32)

    train_model(model,coords,g_codes,epochs)

    return jsonify(large_dataset), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000,threaded=True, debug=True)

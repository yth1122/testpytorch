import torch
import torch.nn as nn
import torch.optim as optim
import os
import re
import math

class CNCModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNCModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
    # 입력 차원 확인 및 조정 
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0) 
    
    
        lstm_out, _ = self.lstm(x)
        x = torch.relu(self.fc1(lstm_out[:, -1, :]))
        return self.fc2(x)

# 모델 저장
def save_model(model, path='/app/data/cnc_model.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': model.lstm.input_size,
        'hidden_size': model.lstm.hidden_size,
        'output_size': model.fc2.out_features
    }, path)
    print(f"Model saved to {path}")

# 모델 불러오기
def load_model(path='/app/data/cnc_model.pth'):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        input_size = checkpoint['input_size']
        hidden_size = checkpoint['hidden_size']
        output_size = checkpoint['output_size']
        
        model = CNCModel(input_size, hidden_size, output_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
        return model
    else:
        print(f"No saved model found at {path}")
        return None

def encode_g_codes(g_codes):
    encoded = torch.zeros((len(g_codes), 3))

    for i, code in enumerate(g_codes):
        # G-code 유형 확인 (G00 또는 G01)
        if code.startswith('G00'):
            encoded[i, 0] = 0
        elif code.startswith('G01'):
            encoded[i, 0] = 1
        else:
            raise ValueError(f"Unsupported G-code: {code}")
        
        # 문자열에서 X와 Z 값 추출
        x_match = re.search(r'X([-]?\d+(\.\d+)?)', code)
        z_match = re.search(r'Z([-]?\d+(\.\d+)?)', code)
        if x_match:
            encoded[i, 1] = float(x_match.group(1))
        if z_match:
            encoded[i, 2] = float(z_match.group(1))
    return encoded
# 교육시키는 function
def train_model(model, coords, g_codes, epochs=1000):
    # 평균 제곱 오차 손실을위해 선언
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    # epoch 반복횟수
    for epoch in range(epochs):
        # optimizer 리셋 전 계산을 제거하기위함
        optimizer.zero_grad()
        outputs = model(coords)
        # 모델에서 가져온값과 g_code 파싱 값과 비교 후 손실 계산 
        loss = criterion(outputs, encode_g_codes(g_codes))
        # 손실에대한 보정
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    save_model(model)

# G코드를 직접 계산하여 보여줄 데이터
def generate_g_code(coords):
    print("gete",coords)
    x_move = coords[1] - coords[0]
    z_move = coords[3] - coords[2]
    if x_move != 0 and z_move != 0:
        return f"G01X{x_move:.1f}Z{z_move:.1f}"
    elif x_move != 0:
        return f"G01X{x_move:.1f}"
    elif z_move != 0:
        return f"G01Z{z_move:.1f}"
    else:
        return "No movement"

def predict_g_code(model, coords, load):
    with torch.no_grad():
        gCodeList = {0:'G00',1:'G01'}
        output = model(coords)
        check = torch.round(output[0, 0])
        # code = gCodeList[check]
        x_axis = torch.round(output[0,1]).to(torch.int)
        z_axis = torch.round(output[0,2]).to(torch.int)
        print("model",output)
        
        if x_axis != 0 and z_axis != 0:
            predicted = f"G01X{x_axis:.1f}Z{z_axis:.1f}"
        elif x_axis != 0:
            predicted = f"G01X{x_axis:.1f}"
        elif z_axis != 0:
            predicted = f"G01Z{z_axis:.1f}"
        else:
            predicted = "No movement"
        
        
    
    actual = generate_g_code(coords)
    return predicted, actual

# 모델 찾기 없으면 생성
def create_or_load_model(input_size=4, hidden_size=64, output_size=3, path='/app/data/cnc_model.pth'):
    model = load_model(path)
    if model is None:
        print(f"Creating new model with input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")
        model = CNCModel(input_size, hidden_size, output_size)
    return model
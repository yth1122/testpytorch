import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 데이터 로드 (예시)
# X = np.array([[x, y, z, load, spindle] for ... in your_data])
# y = np.array([gcode for ... in your_data])

# 데이터 전처리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. MLP 모델
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)
mlp_pred = mlp.predict(X_test_scaled)
print("MLP Accuracy:", accuracy_score(y_test, mlp_pred))
print(classification_report(y_test, mlp_pred))

# 2. Random Forest 모델
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# 3. SVM 모델
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# 특성 중요도 (Random Forest)
feature_importance = rf.feature_importances_
for i, importance in enumerate(feature_importance):
    print(f"Feature {i+1} importance: {importance}")

# 새로운 데이터에 대한 예측
# new_data = np.array([[x, y, z, load, spindle]])
# new_data_scaled = scaler.transform(new_data)
# mlp_prediction = mlp.predict(new_data_scaled)
# rf_prediction = rf.predict(new_data)
# svm_prediction = svm.predict(new_data_scaled)
# print("MLP Prediction:", mlp_prediction)
# print("Random Forest Prediction:", rf_prediction)
# print("SVM Prediction:", svm_prediction)
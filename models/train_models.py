import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# Crear carpeta models si no existe
os.makedirs('models', exist_ok=True)

# 1. Cargar dataset
df = pd.read_excel('../FGR_dataset.xlsx')

# 2. Separar características y etiquetas
X = df.drop(columns=['C31'])  # C31 es la etiqueta: Fetal Weight (0=Normal, 1=FGR)
y = df['C31']

# 3. Escalamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'models/scaler.pkl')

# 4. División de datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -----------------------------
# MODELO 1: REGRESIÓN LOGÍSTICA
# -----------------------------
print("Entrenando Regresión Logística...")
lr = LogisticRegression(max_iter=1000)
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}
grid_lr = GridSearchCV(lr, param_grid_lr, cv=5)
grid_lr.fit(X_train, y_train)
lr_best = grid_lr.best_estimator_
joblib.dump(lr_best, 'models/regression.pkl')

# -----------------------------
# MODELO 2: SVM
# -----------------------------
print("Entrenando SVM...")
svm = SVC(probability=True)
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}
grid_svm = GridSearchCV(svm, param_grid_svm, cv=5)
grid_svm.fit(X_train, y_train)
svm_best = grid_svm.best_estimator_
joblib.dump(svm_best, 'models/svm.pkl')

# -----------------------------
# MODELO 3: RED NEURONAL (ANN)
# -----------------------------
print("Entrenando ANN...")
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
ann_model.save('models/ann.h5')

# -----------------------------
# MODELO 4: MAPA COGNITIVO DIFUSO (simulado)
# -----------------------------
print("Guardando modelo FCM...")
def fuzzy_predict(X):
    """Ejemplo simple: si edad > 35 y presión sistólica > 150 => predice FGR"""
    return np.where((X[:, 0] > 35) & (X[:, 17] > 150), 1, 0)

joblib.dump(fuzzy_predict, 'models/fcm_model.pkl')

# -----------------------------
# EVALUACIÓN FINAL
# -----------------------------
print("\n✅ Entrenamiento finalizado. Evaluaciones:")
for name, model in zip(['Regresión Logística', 'SVM'],
                       [lr_best, svm_best]):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'{name}: {acc:.4f}')

ann_pred = (ann_model.predict(X_test) > 0.5).astype(int)
ann_acc = accuracy_score(y_test, ann_pred)
print(f'ANN: {ann_acc:.4f}')

fcm_pred = fuzzy_predict(X_test)
fcm_acc = accuracy_score(y_test, fcm_pred)
print(f'FCM: {fcm_acc:.4f}')




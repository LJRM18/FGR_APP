from flask import Flask, render_template, request, redirect, url_for, flash
import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import uuid

# Importar FCM desde archivo externo (la versión mejorada que definimos)
from models.fcm_model import fuzzy_predict

# --------------------------
# CONFIGURACIÓN DE LA APP
# --------------------------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------
# CARGAR SCALER Y MODELOS
# -----------------------
scaler = joblib.load('models/scaler.pkl')

models = {
    'regression': joblib.load('models/regression.pkl'),
    'svm': joblib.load('models/svm.pkl'),
    'ann': load_model('models/ann.h5'),
    'fcm': fuzzy_predict  # FCM ahora viene del módulo externo mejorado
}

# -----------------------------
# FUNCIONES DE UTILIDAD
# -----------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Diccionario de validación: (min, max)
validation_ranges = {
    'C1': (10, 55),    'C2': (10, 60),    'C3': (20, 42),    'C4': (0, 10),
    'C5': (0, 10),     'C6': (0, 2),      'C7': (20, 42),    'C8': (0, 100),
    'C9': (20, 42),    'C10': (0, 100),   'C11': (20, 42),   'C12': (0, 100),
    'C13': (20, 42),   'C14': (0, 100),   'C15': (0, 1),     'C16': (0, 1),
    'C17': (0, 2),     'C18': (80, 250),  'C19': (40, 150),  'C20': (0, 5),
    'C21': (0, 1),     'C22': (0, 1000),  'C23': (0, 100),   'C24': (0, 100),
    'C25': (0, 100),   'C26': (0, 100),   'C27': (0, 100),   'C28': (0, 100),
    'C29': (0, 100),   'C30': (0, 1000000)
}

# -----------------------------
# RUTAS
# -----------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_model', methods=['POST'])
def select_model():
    model = request.form['model']
    option = request.form['option']
    if option == 'single':
        return redirect(url_for('predict_single', model=model))
    else:
        return redirect(url_for('predict_batch', model=model))

@app.route('/predict_single/<model>', methods=['GET', 'POST'])
def predict_single(model):
    if request.method == 'POST':
        try:
            # Validar y recopilar inputs
            inputs = []
            for i in range(1, 31):
                key = f'C{i}'
                val_str = request.form.get(key, '').strip()
                if val_str == '':
                    raise ValueError(f'El campo {key} es obligatorio.')

                # Intentar convertir a float
                try:
                    val = float(val_str)
                except:
                    raise ValueError(f'El campo {key} debe ser un número válido.')

                # Validar rango
                min_val, max_val = validation_ranges[key]
                if not (min_val <= val <= max_val):
                    raise ValueError(f'El campo {key} debe estar entre {min_val} y {max_val}.')

                inputs.append(val)

            # Escalar y predecir
            X_input = scaler.transform([inputs])

            if model == 'ann':
                prediction = models[model].predict(X_input)[0][0]
                result = 'FGR' if prediction > 0.5 else 'Normal'
            elif model == 'fcm':
                prediction = models[model](X_input)
                result = 'FGR' if prediction[0] == 1 else 'Normal'
            else:
                prediction = models[model].predict(X_input)[0]
                result = 'FGR' if prediction == 1 else 'Normal'

            return render_template('predict_single.html', model=model, result=result)

        except Exception as e:
            flash(f'Error en la entrada: {e}')
            return redirect(request.url)

    return render_template('predict_single.html', model=model)

@app.route('/predict_batch/<model>', methods=['GET', 'POST'])
def predict_batch(model):
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            flash('Archivo inválido. Solo se permiten .csv o .xlsx.')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            expected_features = [f'C{i}' for i in range(1, 31)]
            missing_cols = [col for col in expected_features if col not in df.columns]

            if missing_cols:
                flash(f"El archivo debe contener exactamente las columnas C1 a C30. Faltan: {', '.join(missing_cols)}.")
                return redirect(request.url)

            X = df[expected_features].values
            y_true = df['C31'].values if 'C31' in df.columns else None
            X_scaled = scaler.transform(X)

            if model == 'ann':
                y_pred = (models[model].predict(X_scaled) > 0.5).astype(int)
            elif model == 'fcm':
                y_pred = models[model](X_scaled)
            else:
                y_pred = models[model].predict(X_scaled)

            acc = accuracy_score(y_true, y_pred) if y_true is not None else None

            if y_true is not None:
                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                img_name = f"static/cm_{uuid.uuid4().hex}.png"
                plt.savefig(img_name)
                plt.close()
            else:
                img_name = None

            return render_template('predict_batch.html', model=model, acc=acc, cm_image=img_name)

        except Exception as e:
            flash(f'Error al procesar el archivo: {e}')
            return redirect(request.url)

    return render_template('predict_batch.html', model=model, acc=None, cm_image=None)

# ---------------------------
# INICIO DEL SERVIDOR
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True)





from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load all models once at startup ──────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), 'models')

with open(f'{BASE}/results_df.pkl',          'rb') as f: results_df          = pickle.load(f)
with open(f'{BASE}/df_pd.pkl',               'rb') as f: df_pd               = pickle.load(f)
with open(f'{BASE}/feature_importances.pkl', 'rb') as f: feature_importances = pickle.load(f)
with open(f'{BASE}/preds_dict.pkl',          'rb') as f: preds_dict          = pickle.load(f)
with open(f'{BASE}/lr_model.pkl',            'rb') as f: lr_data             = pickle.load(f)

feature_columns = lr_data['feature_columns']
lr_coefficients = lr_data['coefficients']
lr_intercept    = lr_data['intercept']
scaler_mean     = lr_data['scaler_mean']
scaler_std      = lr_data['scaler_std']

# Top features for home page stats
feature_names = feature_columns
fi_pairs = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)
top_features = [(name, round(float(val)*100, 1)) for name, val in fi_pairs[:5]]

# Best model
best_idx  = results_df['Accuracy'].idxmax()
best_model = {
    'name':      results_df.loc[best_idx, 'Model'],
    'accuracy':  results_df.loc[best_idx, 'Accuracy'],
    'auc':       results_df.loc[best_idx, 'AUC-ROC'],
    'f1':        results_df.loc[best_idx, 'F1-Score'],
}


def predict_patient(patient_info):
    x = np.array([float(patient_info[c]) for c in feature_columns])
    std_safe = np.where(scaler_std == 0, 1.0, scaler_std)
    x_scaled = (x - scaler_mean) / std_safe
    logit    = np.dot(lr_coefficients, x_scaled) + lr_intercept
    prob_1   = 1.0 / (1.0 + np.exp(-logit))
    prob_0   = 1.0 - prob_1
    pred     = 1 if prob_1 >= 0.25 else 0
    return pred, round(float(prob_1) * 100, 1)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    stats = {
        'total_records': f"{len(df_pd):,}",
        'features':      len(feature_columns),
        'best_accuracy': best_model['accuracy'],
        'best_model':    best_model['name'],
    }
    return render_template('home.html', stats=stats, top_features=top_features)


@app.route('/models')
def models():
    models_data = results_df.to_dict(orient='records')
    return render_template('models.html', models=models_data, best=best_model)


@app.route('/risk')
def risk():
    return render_template('risk.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    age_map = {1: 1, 2: 3, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13}
    age_val = age_map[int(data['age'])]
    bmi     = float(data['bmi'])
    gen_hlth = int(data['gen_hlth'])
    high_bp  = int(data['high_bp'])
    smoker   = int(data['smoker'])
    diabetes = int(data['diabetes'])

    patient = {
        'HighBP':            high_bp,
        'HighChol':          int(data['high_chol']),
        'CholCheck':         1,
        'BMI':               bmi,
        'Smoker':            smoker,
        'Stroke':            1 if (high_bp == 1 and age_val >= 9) else 0,
        'Diabetes':          diabetes,
        'PhysActivity':      0 if gen_hlth >= 4 else 1,
        'Fruits':            1,
        'Veggies':           1,
        'HvyAlcoholConsump': 0,
        'AnyHealthcare':     1,
        'NoDocbcCost':       0,
        'GenHlth':           gen_hlth,
        'MentHlth':          5 if gen_hlth >= 4 else 0,
        'PhysHlth':          10 if gen_hlth >= 4 else 0,
        'DiffWalk':          1 if bmi >= 35 else 0,
        'Sex':               int(data['sex']),
        'Age':               age_val,
        'Education':         5,
        'Income':            5,
    }

    pred, risk_score = predict_patient(patient)

    risk_factors = sum([
        high_bp == 1,
        int(data['high_chol']) == 1,
        bmi >= 30,
        smoker == 1,
        diabetes >= 1,
        gen_hlth >= 4,
        age_val >= 9,
    ])

    if pred == 1 or risk_factors >= 4:
        level = 'high'
        recs  = []
        if high_bp == 1:              recs.append("Monitor blood pressure daily")
        if int(data['high_chol'])==1: recs.append("Follow a low-cholesterol diet")
        if bmi >= 25:                 recs.append("Work towards a healthy BMI")
        if smoker == 1:               recs.append("Consider quitting smoking")
        if diabetes >= 1:             recs.append("Manage blood sugar regularly")
        if gen_hlth >= 4:             recs.append("Seek medical attention soon")
        recs.append("Consult a cardiologist")
    elif risk_factors >= 2 or risk_score >= 15:
        level = 'moderate'
        recs  = [
            "Schedule a health checkup soon",
            "Monitor blood pressure regularly",
            "Exercise at least 30 minutes per day",
        ]
        if smoker == 1: recs.append("Strongly consider quitting smoking")
    else:
        level = 'low'
        recs  = [
            "Keep up your healthy habits",
            "Annual health checkup is advised",
            "Maintain a balanced diet and regular exercise",
        ]

    return jsonify({
        'level':        level,
        'risk_score':   risk_score,
        'risk_factors': risk_factors,
        'recs':         recs,
    })


if __name__ == '__main__':
    app.run(debug=True)

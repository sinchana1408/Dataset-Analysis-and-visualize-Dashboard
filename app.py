from flask import Flask, render_template, request, send_file
import pandas as pd
import os
import numpy as np

# 🔥 IMPORTANT FIX FOR FLASK + MATPLOTLIB
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import *

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

df = None
latest_metrics = {}
show_confusion = False
show_feature = False


# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template('index.html')


# ---------------- UPLOAD ----------------
@app.route('/upload', methods=['POST'])
def upload():
    global df
    file = request.files['file']

    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    df = pd.read_csv(path)

    return render_template('index.html',
                           preview=df.head().to_html(),
                           columns=df.columns)


# ---------------- ANALYZE ----------------
@app.route('/analyze', methods=['POST'])
def analyze():
    global df, latest_metrics, show_confusion, show_feature

    latest_metrics = {}
    show_confusion = False
    show_feature = False

    target = request.form['target']

    X = df.drop(columns=[target])
    y = df[target]

    # Remove ID-like columns (unique columns)
    for col in list(X.columns):
        if X[col].nunique() == len(X):
            X = X.drop(columns=[col])

    # Handle missing values
    X = X.fillna(X.mode().iloc[0])
    y = y.fillna(y.mode()[0])

    # Encode categorical features
    X = pd.get_dummies(X)

    # Encode target if categorical
    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    results = {}

    # Detect task type
    if len(np.unique(y)) <= 10:
        task = "Classification"
        models = {
            "Random Forest": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000)
        }
    else:
        task = "Regression"
        models = {
            "Random Forest": RandomForestRegressor(),
            "Linear Regression": LinearRegression()
        }

    # Train and compare models
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        if task == "Classification":
            results[name] = round(accuracy_score(y_test, pred) * 100, 2)
        else:
            results[name] = round(r2_score(y_test, pred), 2)

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    # -------- METRICS --------

    # Regression metrics
    try:
        latest_metrics["MAE"] = round(mean_absolute_error(y_test, predictions), 2)
        latest_metrics["RMSE"] = round(np.sqrt(mean_squared_error(y_test, predictions)), 2)
        latest_metrics["R2 Score"] = round(r2_score(y_test, predictions), 2)
    except:
        pass

    # Classification metrics
    try:
        latest_metrics["Accuracy (%)"] = round(accuracy_score(y_test, predictions) * 100, 2)
        latest_metrics["Precision (%)"] = round(precision_score(y_test, predictions, average='weighted', zero_division=0) * 100, 2)
        latest_metrics["Recall (%)"] = round(recall_score(y_test, predictions, average='weighted', zero_division=0) * 100, 2)
        latest_metrics["F1 Score (%)"] = round(f1_score(y_test, predictions, average='weighted', zero_division=0) * 100, 2)

        cm = confusion_matrix(y_test, predictions)
        plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.savefig("static/confusion.png")
        plt.close()
        show_confusion = True
    except:
        pass

    # Feature Importance
    if hasattr(best_model, "feature_importances_"):
        plt.figure()
        plt.bar(range(len(best_model.feature_importances_)),
                best_model.feature_importances_)
        plt.title("Feature Importance")
        plt.savefig("static/feature.png")
        plt.close()
        show_feature = True

    return render_template('index.html',
                       preview=df.head().to_html(),
                       columns=df.columns,
                       selected_target=target,
                       task=task,
                       results=results,
                       best=best_model_name,
                       metrics=latest_metrics,
                       show_confusion=show_confusion,
                       show_feature=show_feature)

# ---------------- DOWNLOAD REPORT ----------------
@app.route('/download')
def download():
    global latest_metrics

    doc = SimpleDocTemplate("static/report.pdf")
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AI Analytics Pro - Evaluation Report", styles['Title']))
    elements.append(Spacer(1, 12))

    for key, value in latest_metrics.items():
        elements.append(Paragraph(f"{key} : {value}", styles['Normal']))
        elements.append(Spacer(1, 8))

    doc.build(elements)

    return send_file("static/report.pdf", as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
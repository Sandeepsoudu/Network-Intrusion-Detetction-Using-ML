from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

app = Flask(__name__)
app.config['CSV_UPLOAD_FOLDER'] = 'csv_uploads'  # Folder to store CSV files
app.secret_key = "supersecretkey"

os.makedirs(app.config['CSV_UPLOAD_FOLDER'], exist_ok=True)

# Perform logistic regression (supports binary and multiclass classification)
def perform_logistic_regression(csv_file_path):
    try:
        # Read CSV file
        data = pd.read_csv(csv_file_path)

        # Check for missing values
        if data.isnull().values.any():
            flash("CSV file contains missing values. Filling missing values with column mean.")
            data = data.fillna(data.mean(numeric_only=True))

        # Ensure the dataset has at least one feature column and one target column
        if data.shape[1] < 2:
            flash("CSV must contain at least one feature column and one target column.")
            return None, None, None, None, None

        # Separate features (all except the last column) and target (last column)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Convert categorical features to numeric using LabelEncoder
        le = LabelEncoder()
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = le.fit_transform(X[col])

        # Check if the target variable is categorical and encode it
        if y.dtype == 'object':
            y = le.fit_transform(y)

        # Fit the logistic regression model (multinomial for multiclass classification)
        model = LogisticRegression(max_iter=1000, multi_class='auto', solver='lbfgs')
        model.fit(X, y)

        # Predictions and evaluation metrics
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        class_report = classification_report(y, y_pred, output_dict=True)

        # Return coefficients, intercept, and evaluation metrics
        return model.coef_[0], model.intercept_[0], accuracy, conf_matrix, class_report

    except Exception as e:
        flash(f"Error processing CSV file: {e}")
        return None, None, None, None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is present in the POST request
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash("No file selected.")
                return redirect(request.url)
            
            if file and file.filename.endswith('.csv'):
                # Save the CSV file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['CSV_UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Perform logistic regression and get the results
                results = perform_logistic_regression(filepath)

                # Unpack the results
                coef, intercept, accuracy, conf_matrix, class_report = results

                # Render the results if successful
                if coef is not None:
                    return render_template(
                        'logistic_result.html',
                        coef=coef.tolist(),
                        intercept=intercept,
                        accuracy=accuracy,
                        conf_matrix=conf_matrix.tolist(),
                        class_report=class_report
                    )

    return render_template('upload.html')

if __name__ == '__main__':
    print("Starting Flask app for logistic regression...")
    app.run(debug=True)

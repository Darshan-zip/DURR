

from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
import os
import tempfile
from fpdf import FPDF

app = Flask(__name__)

data = pd.read_csv("merged_fan_export.csv")
data = data.dropna(subset=['Pressure, static', 'Actual flow volume air/gas','Nominal rotation speed'])


num_cols = ['Pressure, static', 'Actual flow volume air/gas', 'Rated power', 'Nominal rotation speed']
for col in num_cols:
    data[col] = (
        data[col]
        .astype(str)
        .str.replace(',', '')
        .str.split().str[0]
        .astype(float)
    )

data['Manufacturer'] = data['Manufacturer'].astype(str).str.strip()

label_encoder = LabelEncoder()
data['Manufacturer_encoded'] = label_encoder.fit_transform(data['Manufacturer'])

X = data[['Pressure, static', 'Actual flow volume air/gas', 'Nominal rotation speed']]
y_power = data['Rated power']
y_manu = data['Manufacturer_encoded']

smote = SMOTE(random_state=42)
X_resampled, y_manu_resampled = smote.fit_resample(X, y_manu)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)
X_power_scaled = scaler.fit_transform(X)
  
clf_manu = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
clf_manu.fit(X_scaled, y_manu_resampled)

reg_power = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
reg_power.fit(X_power_scaled,   y_power)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data_in = request.get_json()
    input_vals = [[data_in['pressure'], data_in['flow'], data_in['speed']]]
    scaled_input = scaler.transform(input_vals)

    power = reg_power.predict(scaled_input)[0]
    manu_encoded = clf_manu.predict(scaled_input)[0]
    manufacturer = label_encoder.inverse_transform([manu_encoded])[0]

    return jsonify({
        "power": round(power, 2),
        "manufacturer": manufacturer
    })

@app.route("/report", methods=["POST"])
def report():
    data_in = request.get_json()
    input_vals = np.array([data_in['pressure'], data_in['flow'], data_in['speed']])

    data['distance'] = X.apply(lambda row: np.linalg.norm(row.values - input_vals), axis=1)
    best_match = data.loc[data['distance'].idxmin()]

    pdf_path = generate_pdf_report(best_match, data_in['pressure'], data_in['flow'], data_in['speed'], best_match['Rated power'])
    filename = os.path.basename(pdf_path)

    report_text = (
        f"Closest Match Details:\n"
        f"Manufacturer: {best_match['Manufacturer']}\n"
        f"Pressure: {best_match['Pressure, static']} Pa\n"
        f"Flow: {best_match['Actual flow volume air/gas']} m3/h\n"
        f"Speed: {best_match['Nominal rotation speed']} 1/min\n"
        f"Rated Power: {best_match['Rated power']} kW"
    )

    return jsonify({"report": report_text, "pdf_filename": filename})

@app.route("/download-pdf")
def download_pdf():
    filename = request.args.get("filename")
    temp_dir = tempfile.gettempdir()
    full_path = os.path.join(temp_dir, filename)

    if not os.path.exists(full_path):
        return "File not found", 404

    return send_file(full_path, as_attachment=True)

def generate_pdf_report(fan_data, static_pressure, flow_volume, rotation_speed, rated_power):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    primary_color = (57, 106, 177)
    secondary_color = (44, 62, 80)

    pdf.set_fill_color(*primary_color)
    pdf.rect(0, 0, 210, 30, style='F')
    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(255, 255, 255)
    pdf.text(15, 20, "FAN SELECTION REPORT")

    pdf.set_y(40)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(*secondary_color)
    pdf.cell(0, 10, "SEARCH PARAMETERS", 0, 1, 'L')
    pdf.line(15, 50, 195, 50)

    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.set_y(55)
    col_width = 90
    row_height = 10

    params = [
        ["Rated Power", f"{rated_power} kW"],
        ["Static Pressure", f"{static_pressure} Pa"],
        ["Flow Volume", f"{flow_volume} m³/h"],
        ["Rotation Speed", f"{rotation_speed} 1/min"]
    ]

    for param in params:
        pdf.cell(col_width, row_height, param[0], 1, 0, 'L')
        pdf.cell(col_width, row_height, param[1], 1, 1, 'L')

    pdf.set_y(95)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(*secondary_color)
    pdf.cell(0, 10, "SELECTED FAN DETAILS", 0, 1, 'L')
    pdf.line(15, 105, 195, 105)

    pdf.set_font('Arial', '', 11)
    pdf.set_y(110)

    details = [
        ["Manufacturer", fan_data['Manufacturer']],
        ["Rated Power", f"{fan_data['Rated power']} kW"],
        ["Rotation Speed", f"{fan_data['Nominal rotation speed']} 1/min"],
        ["Static Pressure", f"{fan_data['Pressure, static']} Pa"],
        ["Flow Volume", f"{fan_data['Actual flow volume air/gas']} m³/h"]
    ]

    for detail in details:
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(col_width, row_height, detail[0], 1, 0, 'L')
        pdf.set_font('Arial', '', 11)
        pdf.cell(col_width, row_height, str(detail[1]), 1, 1, 'L')

    pdf.set_y(-20)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, f"Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 0, 'C')
    pdf.set_draw_color(200, 200, 200)
    pdf.rect(5, 5, 200, 287)

    temp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(temp_dir, f"fan_selection_report.pdf")
    pdf.output(pdf_path)
    return pdf_path

if __name__ == "__main__":
    app.run(debug=True)

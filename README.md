# Fan Predictor Web Application

This application predicts the **rated power** and **manufacturer** of a fan based on input parameters such as:

- Static Pressure (Pa)
- Flow Volume (m³/h)
- Nominal Rotation Speed (1/min)

It also provides a **PDF report** of the closest matching fan based on historical data.

---

## 🔧 Technologies Used

- Python (Flask)
- Pandas, NumPy
- Scikit-learn (Gradient Boosting)
- imbalanced-learn (SMOTE)
- FPDF (for PDF generation)
- HTML + Bootstrap (frontend)

---

## 📁 Folder Structure

│
├── data.csv # Your fan dataset
├── main.py # Main Flask backend
├── templates/
│ └── index.html # Frontend HTML interface
└── README.md

#HomePage
![image](https://github.com/user-attachments/assets/a4a72330-7741-4768-a329-2a08af99bc1e)
#Generated Report
![image](https://github.com/user-attachments/assets/20858efe-d9ad-4a2f-bff3-426046c17e5a)


---
Steps to execute the code:
1. Clone the repository
2. Install the necessary dependencies using: pip install flask pandas numpy scikit-learn fpdf imbalanced-learn
3. Place the data.csv file in the project directory
4. Run the main.py file in the terminal using : python main.py

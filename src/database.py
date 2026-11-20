
import sqlite3
from datetime import datetime

# Database file
DATABASE_PATH = "stroke_records.db"


# ==========================================
# 1. INITIALIZE DATABASE
# ==========================================

def initialize_database():

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            created_at TEXT,

            gender TEXT,
            age REAL,
            hypertension INTEGER,
            heart_disease INTEGER,
            ever_married TEXT,
            work_type TEXT,
            residence_type TEXT,
            avg_glucose_level REAL,
            bmi REAL,
            smoking_status TEXT,

            prediction INTEGER,
            probability REAL

        )
    """)

    connection.commit()
    connection.close()

    print("Database initialized successfully!")


# ==========================================
# 2. SAVE PATIENT ASSESSMENT
# ==========================================

def save_assessment(patient, prediction, probability):

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO assessments (

            created_at,
            gender,
            age,
            hypertension,
            heart_disease,
            ever_married,
            work_type,
            residence_type,
            avg_glucose_level,
            bmi,
            smoking_status,
            prediction,
            probability

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

    """, (

        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        patient["gender"],
        patient["age"],
        patient["hypertension"],
        patient["heart_disease"],
        patient["ever_married"],
        patient["work_type"],
        patient["Residence_type"],
        patient["avg_glucose_level"],
        patient["bmi"],
        patient["smoking_status"],

        int(prediction),
        float(probability)

    ))

    connection.commit()
    connection.close()

    print("Assessment saved successfully!")


# ==========================================
# 3. GET ALL ASSESSMENTS
# ==========================================

def get_all_assessments():

    connection = sqlite3.connect(DATABASE_PATH)

    records = connection.execute("""
        SELECT * FROM assessments
        ORDER BY id DESC
    """).fetchall()

    connection.close()

    return records


# ==========================================
# 4. DELETE ALL ASSESSMENTS
# ==========================================

def clear_assessments():

    connection = sqlite3.connect(DATABASE_PATH)

    connection.execute("DELETE FROM assessments")

    connection.commit()
    connection.close()

    print("All assessment records deleted!")


# ==========================================
# 5. INITIALIZE DATABASE
# ==========================================

if __name__ == "__main__":

    initialize_database()

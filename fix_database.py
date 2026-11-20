
import sqlite3

DATABASE = "stroke_records.db"


def fix_database():

    try:
        connection = sqlite3.connect(DATABASE)
        cursor = connection.cursor()

        # Check whether the assessments table exists
        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            AND name='assessments'
        """)

        if cursor.fetchone() is None:
            print("Error: assessments table not found.")
            connection.close()
            return

        # Read existing column names
        cursor.execute("PRAGMA table_info(assessments)")

        columns = [
            column[1]
            for column in cursor.fetchall()
        ]

        print("Existing columns:")
        print(columns)

        # Add the missing date column
        if "date" not in columns:

            cursor.execute("""
                ALTER TABLE assessments
                ADD COLUMN date TEXT
            """)

            print("Successfully added date column.")

            # Copy old timestamps if available
            if "timestamp" in columns:
                cursor.execute("""
                    UPDATE assessments
                    SET date = timestamp
                    WHERE date IS NULL
                """)

            elif "created_at" in columns:
                cursor.execute("""
                    UPDATE assessments
                    SET date = created_at
                    WHERE date IS NULL
                """)

        else:
            print("Date column already exists.")

        connection.commit()

        # Display updated structure
        cursor.execute("PRAGMA table_info(assessments)")

        print("\nUpdated columns:")

        for column in cursor.fetchall():
            print(column[1])

        connection.close()

        print("\nDatabase migration completed successfully!")

    except sqlite3.Error as error:
        print("Database error:", error)


if __name__ == "__main__":
    fix_database()

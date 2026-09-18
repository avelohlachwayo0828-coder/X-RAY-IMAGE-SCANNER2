import sqlite3
from datetime import datetime


DATABASE_NAME = "scans.db"



# Create database and table

def create_database():

    connection = sqlite3.connect(
        DATABASE_NAME
    )

    cursor = connection.cursor()


    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            scan_date TEXT,

            image_name TEXT,

            report TEXT

        )
        """
    )


    connection.commit()

    connection.close()



# Save a scan result

def save_scan(
        image_name,
        report
):

    connection = sqlite3.connect(
        DATABASE_NAME
    )

    cursor = connection.cursor()


    date = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    cursor.execute(
        """
        INSERT INTO scans
        (
            scan_date,
            image_name,
            report
        )

        VALUES
        (?, ?, ?)

        """,

        (
            date,
            image_name,
            report
        )
    )


    connection.commit()

    connection.close()



# Retrieve scan history

def get_scans():

    connection = sqlite3.connect(
        DATABASE_NAME
    )

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *
        FROM scans
        ORDER BY id DESC
        """
    )


    scans = cursor.fetchall()


    connection.close()


    return scans
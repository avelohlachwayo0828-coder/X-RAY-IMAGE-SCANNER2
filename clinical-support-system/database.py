import sqlite3

from pathlib import Path

from werkzeug.security import generate_password_hash


# =========================================================
# DATABASE LOCATION
# =========================================================

BASE_DIR = Path(
    __file__
).resolve().parent

DATABASE = (
    BASE_DIR /
    "hospital.db"
)


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE
    )

    connection.row_factory = (
        sqlite3.Row
    )

    return connection


# =========================================================
# CREATE DATABASE
# =========================================================

def create_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            staff_id TEXT NOT NULL UNIQUE,

            password_hash TEXT NOT NULL,

            role TEXT NOT NULL
                CHECK (
                    role IN (
                        'clinical_user',
                        'technician'
                    )
                ),

            full_name TEXT NOT NULL,

            status TEXT NOT NULL
                DEFAULT 'active'

                CHECK (
                    status IN (
                        'active',
                        'inactive'
                    )
                ),

            created_at
                TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            updated_at
                TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()

    connection.close()


# =========================================================
# CREATE CASES TABLE
# =========================================================

def create_cases_table():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS cases (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            case_id TEXT NOT NULL UNIQUE,

            staff_id TEXT NOT NULL,

            image_filename TEXT NOT NULL,

            status TEXT NOT NULL
                DEFAULT 'pending'

                CHECK (
                    status IN (
                        'pending',
                        'analysed',
                        'failed'
                    )
                ),

            prediction TEXT,

            confidence REAL,

            gradcam_filename TEXT,

            gemini_interpretation TEXT,

            created_at
                TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (staff_id)
                REFERENCES users(staff_id)
        )
        """
    )

    connection.commit()

    connection.close()

    ensure_case_columns()


# =========================================================
# ENSURE CASE COLUMNS
# =========================================================

def ensure_case_columns():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        "PRAGMA table_info(cases)"
    )

    columns = [
        row["name"]
        for row in cursor.fetchall()
    ]

    if (
        "gemini_interpretation"
        not in columns
    ):

        cursor.execute(
            """
            ALTER TABLE cases
            ADD COLUMN gemini_interpretation TEXT
            """
        )

    connection.commit()

    connection.close()


# =========================================================
# INITIAL ACCOUNTS
# =========================================================

def create_initial_users():

    connection = get_connection()

    cursor = connection.cursor()

    users = [

        (
            "STAFF001",
            "Staff123!",
            "clinical_user",
            "Clinical User 1"
        ),

        (
            "STAFF002",
            "Staff456!",
            "clinical_user",
            "Clinical User 2"
        ),

        (
            "TECH001",
            "Tech123!",
            "technician",
            "System Technician 1"
        ),

        (
            "TECH002",
            "Tech456!",
            "technician",
            "System Technician 2"
        )

    ]

    for (
        staff_id,
        password,
        role,
        full_name
    ) in users:

        password_hash = (
            generate_password_hash(
                password
            )
        )

        try:

            cursor.execute(
                """
                INSERT INTO users
                (
                    staff_id,
                    password_hash,
                    role,
                    full_name
                )

                VALUES (?, ?, ?, ?)
                """,
                (
                    staff_id,
                    password_hash,
                    role,
                    full_name
                )
            )

        except sqlite3.IntegrityError:

            pass

    connection.commit()

    connection.close()


# =========================================================
# FIND USER
# =========================================================

def get_user_by_staff_id(
    staff_id
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM users
        WHERE staff_id = ?
        """,
        (
            staff_id,
        )
    )

    user = cursor.fetchone()

    connection.close()

    return user


# =========================================================
# CREATE STAFF ACCOUNT
# =========================================================

def create_staff_account(
    staff_id,
    password,
    full_name
):

    connection = get_connection()

    cursor = connection.cursor()

    password_hash = (
        generate_password_hash(
            password
        )
    )

    try:

        cursor.execute(
            """
            INSERT INTO users
            (
                staff_id,
                password_hash,
                role,
                full_name,
                status
            )

            VALUES (?, ?, ?, ?, ?)
            """,
            (
                staff_id,
                password_hash,
                "clinical_user",
                full_name,
                "active"
            )
        )

        connection.commit()

        connection.close()

        return (
            True,
            "Staff account created successfully."
        )

    except sqlite3.IntegrityError:

        connection.close()

        return (
            False,
            "This Staff ID already exists."
        )


# =========================================================
# GET CLINICAL USERS
# =========================================================

def get_clinical_users():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            staff_id,
            full_name,
            status,
            created_at,
            updated_at

        FROM users

        WHERE role = 'clinical_user'

        ORDER BY id DESC
        """
    )

    users = cursor.fetchall()

    connection.close()

    return users


# =========================================================
# UPDATE STAFF ID
# =========================================================

def update_staff_id(
    user_id,
    new_staff_id
):

    connection = get_connection()

    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            UPDATE users

            SET
                staff_id = ?,
                updated_at = CURRENT_TIMESTAMP

            WHERE
                id = ?

                AND role = 'clinical_user'
            """,
            (
                new_staff_id,
                user_id
            )
        )

        if cursor.rowcount == 0:

            connection.close()

            return (
                False,
                "Staff account not found."
            )

        connection.commit()

        connection.close()

        return (
            True,
            "Staff ID updated successfully."
        )

    except sqlite3.IntegrityError:

        connection.close()

        return (
            False,
            "That Staff ID is already in use."
        )


# =========================================================
# UPDATE PASSWORD
# =========================================================

def update_staff_password(
    user_id,
    new_password
):

    connection = get_connection()

    cursor = connection.cursor()

    password_hash = (
        generate_password_hash(
            new_password
        )
    )

    cursor.execute(
        """
        UPDATE users

        SET
            password_hash = ?,
            updated_at = CURRENT_TIMESTAMP

        WHERE
            id = ?

            AND role = 'clinical_user'
        """,
        (
            password_hash,
            user_id
        )
    )

    if cursor.rowcount == 0:

        connection.close()

        return (
            False,
            "Staff account not found."
        )

    connection.commit()

    connection.close()

    return (
        True,
        "Password updated successfully."
    )


# =========================================================
# UPDATE STAFF NAME
# =========================================================

def update_staff_name(
    user_id,
    new_name
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE users

        SET
            full_name = ?,
            updated_at = CURRENT_TIMESTAMP

        WHERE
            id = ?

            AND role = 'clinical_user'
        """,
        (
            new_name,
            user_id
        )
    )

    if cursor.rowcount == 0:

        connection.close()

        return (
            False,
            "Staff account not found."
        )

    connection.commit()

    connection.close()

    return (
        True,
        "Staff name updated successfully."
    )


# =========================================================
# DEACTIVATE STAFF
# =========================================================

def deactivate_staff(
    user_id
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE users

        SET
            status = 'inactive',
            updated_at = CURRENT_TIMESTAMP

        WHERE
            id = ?

            AND role = 'clinical_user'
        """,
        (
            user_id,
        )
    )

    if cursor.rowcount == 0:

        connection.close()

        return (
            False,
            "Staff account not found."
        )

    connection.commit()

    connection.close()

    return (
        True,
        "Staff account deactivated."
    )


# =========================================================
# ACTIVATE STAFF
# =========================================================

def activate_staff(
    user_id
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE users

        SET
            status = 'active',
            updated_at = CURRENT_TIMESTAMP

        WHERE
            id = ?

            AND role = 'clinical_user'
        """,
        (
            user_id,
        )
    )

    if cursor.rowcount == 0:

        connection.close()

        return (
            False,
            "Staff account not found."
        )

    connection.commit()

    connection.close()

    return (
        True,
        "Staff account activated."
    )


# =========================================================
# CREATE CASE
# =========================================================

def create_case(
    case_id,
    staff_id,
    image_filename
):

    connection = get_connection()

    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO cases
            (
                case_id,
                staff_id,
                image_filename,
                status
            )

            VALUES (?, ?, ?, ?)
            """,
            (
                case_id,
                staff_id,
                image_filename,
                "pending"
            )
        )

        connection.commit()

        connection.close()

        return (
            True,
            "Case created successfully."
        )

    except sqlite3.IntegrityError:

        connection.close()

        return (
            False,
            "Case ID already exists."
        )


# =========================================================
# GET CASES FOR ONE STAFF MEMBER
# =========================================================

def get_cases_by_staff(
    staff_id
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            case_id,
            staff_id,
            image_filename,
            status,
            prediction,
            confidence,
            gradcam_filename,
            gemini_interpretation,
            created_at

        FROM cases

        WHERE staff_id = ?

        ORDER BY created_at DESC
        """,
        (
            staff_id,
        )
    )

    cases = cursor.fetchall()

    connection.close()

    return cases


# =========================================================
# GET ALL CASES
# =========================================================

def get_all_cases():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            case_id,
            staff_id,
            image_filename,
            status,
            prediction,
            confidence,
            gradcam_filename,
            gemini_interpretation,
            created_at

        FROM cases

        ORDER BY created_at DESC
        """
    )

    cases = cursor.fetchall()

    connection.close()

    return cases


# =========================================================
# GET SINGLE CASE
# =========================================================

def get_case(
    case_id
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM cases
        WHERE case_id = ?
        """,
        (
            case_id,
        )
    )

    case = cursor.fetchone()

    connection.close()

    return case


# =========================================================
# UPDATE CASE RESULT
# =========================================================

def update_case_result(
    case_id,
    prediction,
    confidence,
    gradcam_filename=None
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE cases

        SET
            status = 'analysed',
            prediction = ?,
            confidence = ?,
            gradcam_filename = ?

        WHERE
            case_id = ?
        """,
        (
            prediction,
            confidence,
            gradcam_filename,
            case_id
        )
    )

    if cursor.rowcount == 0:

        connection.close()

        return (
            False,
            "Case not found."
        )

    connection.commit()

    connection.close()

    return (
        True,
        "Case result updated successfully."
    )


# =========================================================
# SAVE GEMINI INTERPRETATION
# =========================================================

def save_gemini_interpretation(
    case_id,
    interpretation
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE cases

        SET
            gemini_interpretation = ?

        WHERE
            case_id = ?
        """,
        (
            interpretation,
            case_id
        )
    )

    if cursor.rowcount == 0:

        connection.close()

        return (
            False,
            "Case not found."
        )

    connection.commit()

    connection.close()

    return (
        True,
        "Gemini interpretation saved."
    )


# =========================================================
# GET GEMINI INTERPRETATION
# =========================================================

def get_gemini_interpretation(
    case_id
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            gemini_interpretation

        FROM cases

        WHERE case_id = ?
        """,
        (
            case_id,
        )
    )

    row = cursor.fetchone()

    connection.close()

    if row is None:

        return None

    return row[
        "gemini_interpretation"
    ]


# =========================================================
# UPDATE CASE STATUS
# =========================================================

def update_case_status(
    case_id,
    status
):

    allowed_statuses = [

        "pending",
        "analysed",
        "failed"

    ]

    if status not in allowed_statuses:

        return (
            False,
            "Invalid case status."
        )

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE cases

        SET
            status = ?

        WHERE
            case_id = ?
        """,
        (
            status,
            case_id
        )
    )

    if cursor.rowcount == 0:

        connection.close()

        return (
            False,
            "Case not found."
        )

    connection.commit()

    connection.close()

    return (
        True,
        "Case status updated successfully."
    )


# =========================================================
# GET CASE STATISTICS
# =========================================================

def get_case_statistics(
    staff_id=None
):

    connection = get_connection()

    cursor = connection.cursor()

    if staff_id:

        cursor.execute(
            """
            SELECT

                COUNT(*) AS total_cases,

                COALESCE(
                    SUM(
                        CASE
                            WHEN status = 'analysed'
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS analysed_cases,

                COALESCE(
                    SUM(
                        CASE
                            WHEN status = 'pending'
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS pending_cases,

                COALESCE(
                    SUM(
                        CASE
                            WHEN status = 'failed'
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS failed_cases

            FROM cases

            WHERE staff_id = ?
            """,
            (
                staff_id,
            )
        )

    else:

        cursor.execute(
            """
            SELECT

                COUNT(*) AS total_cases,

                COALESCE(
                    SUM(
                        CASE
                            WHEN status = 'analysed'
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS analysed_cases,

                COALESCE(
                    SUM(
                        CASE
                            WHEN status = 'pending'
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS pending_cases,

                COALESCE(
                    SUM(
                        CASE
                            WHEN status = 'failed'
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS failed_cases

            FROM cases
            """
        )

    statistics = cursor.fetchone()

    connection.close()

    return statistics


# =========================================================
# DATABASE SETUP
# =========================================================

if __name__ == "__main__":

    create_database()

    create_initial_users()

    create_cases_table()

    print()

    print(
        "Hospital database ready."
    )

    print()

    print(
        f"Database: {DATABASE}"
    )

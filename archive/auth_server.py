from pathlib import Path

from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)

from werkzeug.security import (
    check_password_hash
)

from werkzeug.utils import (
    secure_filename
)

from PIL import Image


from database import (
    create_database,
    create_initial_users,
    create_cases_table,

    get_user_by_staff_id,

    create_staff_account,
    get_clinical_users,

    update_staff_id,
    update_staff_password,
    update_staff_name,

    deactivate_staff,
    activate_staff,

    create_case,
    get_all_cases,
    get_case,
    update_case_result,
    update_case_status,
    get_case_statistics
)


from predict import (
    predict_tb
)


from gradcam import (
    generate_gradcam
)


from ai_interpreter import (
    interpret_chest_xray
)

# =========================================================

# =========================================================
# APPLICATION
# =========================================================

app = Flask(
    __name__
)

app.secret_key = (
    "change-this-to-a-long-random-secret-key"
)


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = Path(
    __file__
).resolve().parent


# =========================================================
# UPLOAD DIRECTORY
# =========================================================

UPLOAD_FOLDER = (
    BASE_DIR
    / "static"
    / "uploads"
)

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


app.config[
    "UPLOAD_FOLDER"
] = str(
    UPLOAD_FOLDER
)


# =========================================================
# ALLOWED FILES
# =========================================================

ALLOWED_EXTENSIONS = {

    "jpg",
    "jpeg",
    "png"

}


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

create_database()

create_initial_users()

create_cases_table()


# =========================================================
# FILE VALIDATION
# =========================================================

def allowed_file(filename):

    return (

        "." in filename

        and

        filename.rsplit(
            ".",
            1
        )[1].lower()

        in ALLOWED_EXTENSIONS

    )


# =========================================================
# LOGIN PAGE
# =========================================================

@app.route("/")
def login_page():

    if "user_id" in session:

        if session.get(
            "role"
        ) == "technician":

            return redirect(
                url_for(
                    "technician_dashboard"
                )
            )

        return redirect(
            url_for(
                "clinical_dashboard"
            )
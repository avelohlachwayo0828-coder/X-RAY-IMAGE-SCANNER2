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


from gemini_interpreter import (
    interpret_chest_xray
)


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
        )

    return render_template(
        "index.html"
    )


# =========================================================
# LOGIN
# =========================================================

@app.route(
    "/login",
    methods=["POST"]
)
def login():

    staff_id = request.form.get(
        "staff_id",
        ""
    ).strip().upper()

    password = request.form.get(
        "password",
        ""
    )

    if not staff_id or not password:

        return render_template(
            "index.html",
            error=(
                "Please enter your "
                "Staff ID and password."
            )
        )

    user = get_user_by_staff_id(
        staff_id
    )

    if user is None:

        return render_template(
            "index.html",
            error=(
                "Invalid Staff ID "
                "or password."
            )
        )

    if user["status"] != "active":

        return render_template(
            "index.html",
            error=(
                "This account is inactive. "
                "Contact the technician."
            )
        )

    if not check_password_hash(
        user["password_hash"],
        password
    ):

        return render_template(
            "index.html",
            error=(
                "Invalid Staff ID "
                "or password."
            )
        )

    session.clear()

    session["user_id"] = user["id"]

    session["staff_id"] = (
        user["staff_id"]
    )

    session["full_name"] = (
        user["full_name"]
    )

    session["role"] = (
        user["role"]
    )

    if user["role"] == "technician":

        return redirect(
            url_for(
                "technician_dashboard"
            )
        )

    return redirect(
        url_for(
            "clinical_dashboard"
        )
    )


# =========================================================
# CLINICAL DASHBOARD
# =========================================================

@app.route(
    "/clinical-dashboard"
)
def clinical_dashboard():

    if "user_id" not in session:

        return redirect(
            url_for(
                "login_page"
            )
        )

    if session.get(
        "role"
    ) != "clinical_user":

        return redirect(
            url_for(
                "login_page"
            )
        )

    cases = get_all_cases()

    statistics = get_case_statistics()

    return render_template(

        "clinical_dashboard.html",

        full_name=session[
            "full_name"
        ],

        staff_id=session[
            "staff_id"
        ],

        cases=cases,

        statistics=statistics

    )


# =========================================================
# NEW SCAN
# =========================================================

@app.route(
    "/new-scan",
    methods=["POST"]
)
def new_scan():

    if "user_id" not in session:

        return redirect(
            url_for(
                "login_page"
            )
        )

    if session.get(
        "role"
    ) != "clinical_user":

        return redirect(
            url_for(
                "login_page"
            )
        )

    # -----------------------------------------------------
    # CHECK FILE
    # -----------------------------------------------------

    if "xray" not in request.files:

        flash(
            "Please select an X-ray image.",
            "error"
        )

        return redirect(
            url_for(
                "clinical_dashboard"
            )
        )

    file = request.files[
        "xray"
    ]

    if file.filename == "":

        flash(
            "Please select an X-ray image.",
            "error"
        )

        return redirect(
            url_for(
                "clinical_dashboard"
            )
        )

    if not allowed_file(
        file.filename
    ):

        flash(
            (
                "Only JPG, JPEG and PNG "
                "images are allowed."
            ),
            "error"
        )

        return redirect(
            url_for(
                "clinical_dashboard"
            )
        )

    # -----------------------------------------------------
    # CREATE CASE ID
    # -----------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d%H%M%S%f"
    )

    case_id = (
        f"CASE-{timestamp}"
    )

    # -----------------------------------------------------
    # SAFE FILE NAME
    # -----------------------------------------------------

    original_filename = (
        secure_filename(
            file.filename
        )
    )

    saved_filename = (
        f"{case_id}_{original_filename}"
    )

    file_path = (
        UPLOAD_FOLDER
        / saved_filename
    )

    # -----------------------------------------------------
    # SAVE IMAGE
    # -----------------------------------------------------

    try:

        file.save(
            file_path
        )

    except Exception as error:

        print(
            "IMAGE SAVE ERROR:",
            error
        )

        flash(
            "Unable to save the X-ray image.",
            "error"
        )

        return redirect(
            url_for(
                "clinical_dashboard"
            )
        )

    # -----------------------------------------------------
    # CREATE DATABASE CASE
    # -----------------------------------------------------

    success, message = create_case(

        case_id,

        session["staff_id"],

        saved_filename

    )

    if not success:

        if file_path.exists():

            file_path.unlink()

        flash(
            message,
            "error"
        )

        return redirect(
            url_for(
                "clinical_dashboard"
            )
        )

    # -----------------------------------------------------
    # RUN ANALYSIS
    # -----------------------------------------------------

    try:

        image = Image.open(
            file_path
        )

        image.load()

        # ---------------------------------------------
        # PREDICTION
        # ---------------------------------------------

        result = predict_tb(
            image
        )

        print(
            "\nMODEL PREDICTION:"
        )

        print(
            result
        )

        # ---------------------------------------------
        # GRAD-CAM
        # ---------------------------------------------

        heatmap = generate_gradcam(
            image
        )

        gradcam_filename = (
            f"{case_id}_heatmap.jpg"
        )

        gradcam_path = (
            UPLOAD_FOLDER
            / gradcam_filename
        )

        heatmap_image = Image.fromarray(
            heatmap
        )

        heatmap_image.save(
            gradcam_path,
            "JPEG",
            quality=95
        )

        # ---------------------------------------------
        # GEMINI IMAGE EXPLANATION
        # ---------------------------------------------

        gemini_text = (
            interpret_chest_xray(
                image
            )
        )

        print(
            "\nGEMINI IMAGE EXPLANATION:"
        )

        print(
            gemini_text
        )

        # ---------------------------------------------
        # SAVE GEMINI EXPLANATION
        #
        # We use a separate text file so the database
        # structure does not need to change.
        # ---------------------------------------------

        gemini_filename = (
            f"{case_id}_ai.txt"
        )

        gemini_path = (
            UPLOAD_FOLDER
            / gemini_filename
        )

        gemini_path.write_text(
            gemini_text,
            encoding="utf-8"
        )

        # ---------------------------------------------
        # UPDATE DATABASE
        # ---------------------------------------------

        update_case_result(

            case_id,

            result["diagnosis"],

            result["confidence"],

            gradcam_filename

        )

    except Exception as error:

        print(
            "\nANALYSIS ERROR:",
            error
        )

        update_case_status(
            case_id,
            "failed"
        )

        flash(
            (
                "The image was uploaded, "
                "but analysis could not be completed."
            ),
            "error"
        )

        return redirect(
            url_for(
                "clinical_dashboard"
            )
        )

    # -----------------------------------------------------
    # ANALYSIS COMPLETE
    # -----------------------------------------------------

    return redirect(
        url_for(
            "case_details",
            case_id=case_id
        )
    )


# =========================================================
# CASE DETAILS
# =========================================================

@app.route(
    "/case/<case_id>"
)
def case_details(case_id):

    if "user_id" not in session:

        return redirect(
            url_for(
                "login_page"
            )
        )

    if session.get(
        "role"
    ) != "clinical_user":

        return redirect(
            url_for(
                "login_page"
            )
        )

    case = get_case(
        case_id
    )

    if case is None:

        flash(
            "Case not found.",
            "error"
        )

        return redirect(
            url_for(
                "clinical_dashboard"
            )
        )

    # -----------------------------------------------------
    # LOAD GEMINI EXPLANATION
    # -----------------------------------------------------

    gemini_filename = (
        f"{case_id}_ai.txt"
    )

    gemini_path = (
        UPLOAD_FOLDER
        / gemini_filename
    )

    gemini_interpretation = None

    if gemini_path.exists():

        try:

            gemini_interpretation = (
                gemini_path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception as error:

            print(
                "GEMINI TEXT READ ERROR:",
                error
            )

            gemini_interpretation = None

    # -----------------------------------------------------
    # RENDER CASE
    # -----------------------------------------------------

    return render_template(

        "case_details.html",

        case=case,

        gemini_interpretation=(
            gemini_interpretation
        ),

        full_name=session[
            "full_name"
        ],

        staff_id=session[
            "staff_id"
        ]

    )


# =========================================================
# TECHNICIAN DASHBOARD
# =========================================================

@app.route(
    "/technician-dashboard"
)
def technician_dashboard():

    if not technician_required():

        return redirect(
            url_for(
                "login_page"
            )
        )

    users = get_clinical_users()

    cases = get_all_cases()

    statistics = get_case_statistics()

    return render_template(

        "technician_dashboard.html",

        users=users,

        cases=cases,

        statistics=statistics,

        full_name=session[
            "full_name"
        ],

        staff_id=session[
            "staff_id"
        ]

    )


# =========================================================
# CREATE STAFF
# =========================================================

@app.route(
    "/create-staff",
    methods=["POST"]
)
def create_staff():

    if not technician_required():

        return redirect(
            url_for(
                "login_page"
            )
        )

    full_name = request.form.get(
        "full_name",
        ""
    ).strip()

    staff_id = request.form.get(
        "staff_id",
        ""
    ).strip().upper()

    password = request.form.get(
        "password",
        ""
    )

    if (
        not full_name
        or not staff_id
        or not password
    ):

        return technician_message(
            error="All fields are required."
        )

    if not staff_id.startswith(
        "STAFF"
    ):

        return technician_message(
            error=(
                "Staff IDs must start "
                "with STAFF."
            )
        )

    if len(password) < 8:

        return technician_message(
            error=(
                "Password must contain "
                "at least 8 characters."
            )
        )

    success, message = (
        create_staff_account(
            staff_id,
            password,
            full_name
        )
    )

    if success:

        return technician_message(
            success=message
        )

    return technician_message(
        error=message
    )


# =========================================================
# UPDATE STAFF ID
# =========================================================

@app.route(
    "/update-staff-id",
    methods=["POST"]
)
def update_staff_id_route():

    if not technician_required():

        return redirect(
            url_for(
                "login_page"
            )
        )

    user_id = request.form.get(
        "user_id",
        ""
    )

    new_staff_id = request.form.get(
        "new_staff_id",
        ""
    ).strip().upper()

    if (
        not user_id
        or not new_staff_id
    ):

        return technician_message(
            error=(
                "Please provide a "
                "new Staff ID."
            )
        )

    if not new_staff_id.startswith(
        "STAFF"
    ):

        return technician_message(
            error=(
                "Staff IDs must start "
                "with STAFF."
            )
        )

    success, message = (
        update_staff_id(
            user_id,
            new_staff_id
        )
    )

    if success:

        return technician_message(
            success=message
        )

    return technician_message(
        error=message
    )


# =========================================================
# UPDATE PASSWORD
# =========================================================

@app.route(
    "/update-staff-password",
    methods=["POST"]
)
def update_staff_password_route():

    if not technician_required():

        return redirect(
            url_for(
                "login_page"
            )
        )

    user_id = request.form.get(
        "user_id",
        ""
    )

    new_password = request.form.get(
        "new_password",
        ""
    )

    if (
        not user_id
        or not new_password
    ):

        return technician_message(
            error=(
                "Please provide a "
                "new password."
            )
        )

    if len(new_password) < 8:

        return technician_message(
            error=(
                "Password must contain "
                "at least 8 characters."
            )
        )

    success, message = (
        update_staff_password(
            user_id,
            new_password
        )
    )

    if success:

        return technician_message(
            success=message
        )

    return technician_message(
        error=message
    )


# =========================================================
# UPDATE STAFF NAME
# =========================================================

@app.route(
    "/update-staff-name",
    methods=["POST"]
)
def update_staff_name_route():

    if not technician_required():

        return redirect(
            url_for(
                "login_page"
            )
        )

    user_id = request.form.get(
        "user_id",
        ""
    )

    new_name = request.form.get(
        "new_name",
        ""
    ).strip()

    if (
        not user_id
        or not new_name
    ):

        return technician_message(
            error="Please provide a name."
        )

    success, message = (
        update_staff_name(
            user_id,
            new_name
        )
    )

    if success:

        return technician_message(
            success=message
        )

    return technician_message(
        error=message
    )


# =========================================================
# DEACTIVATE STAFF
# =========================================================

@app.route(
    "/deactivate-staff",
    methods=["POST"]
)
def deactivate_staff_route():

    if not technician_required():

        return redirect(
            url_for(
                "login_page"
            )
        )

    user_id = request.form.get(
        "user_id",
        ""
    )

    if not user_id:

        return technician_message(
            error="Invalid staff account."
        )

    success, message = (
        deactivate_staff(
            user_id
        )
    )

    if success:

        return technician_message(
            success=message
        )

    return technician_message(
        error=message
    )


# =========================================================
# ACTIVATE STAFF
# =========================================================

@app.route(
    "/activate-staff",
    methods=["POST"]
)
def activate_staff_route():

    if not technician_required():

        return redirect(
            url_for(
                "login_page"
            )
        )

    user_id = request.form.get(
        "user_id",
        ""
    )

    if not user_id:

        return technician_message(
            error="Invalid staff account."
        )

    success, message = (
        activate_staff(
            user_id
        )
    )

    if success:

        return technician_message(
            success=message
        )

    return technician_message(
        error=message
    )


# =========================================================
# TECHNICIAN CHECK
# =========================================================

def technician_required():

    if "user_id" not in session:

        return False

    if session.get(
        "role"
    ) != "technician":

        return False

    return True


# =========================================================
# TECHNICIAN MESSAGE
# =========================================================

def technician_message(
    success=None,
    error=None
):

    return render_template(

        "technician_dashboard.html",

        users=get_clinical_users(),

        cases=get_all_cases(),

        statistics=get_case_statistics(),

        full_name=session[
            "full_name"
        ],

        staff_id=session[
            "staff_id"
        ],

        success=success,

        error=error

    )


# =========================================================
# LOGOUT
# =========================================================

@app.route(
    "/logout"
)
def logout():

    session.clear()

    return redirect(
        url_for(
            "login_page"
        )
    )


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )
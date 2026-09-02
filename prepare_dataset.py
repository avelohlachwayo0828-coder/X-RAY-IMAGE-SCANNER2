from pathlib import Path
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    PROJECT_DIR
    / "dataset"
    / "Chest X_Ray Dataset"
)

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = RESULTS_DIR / "master_df.csv"
DUPLICATES_PATH = RESULTS_DIR / "duplicates_df.csv"
SPLIT_PATH = RESULTS_DIR / "split_data_verified.csv"

RANDOM_STATE = 42

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

CLASS_MAPPING = {
    "COVID19": "COVID19",
    "NORMAL": "NORMAL",
    "PNEUMONIA": "PNEUMONIA",
    "TB": "TB",
    "TURBERCULOSIS": "TB",
}


# ============================================================
# DATASET CHECK
# ============================================================

print("=" * 60)
print("STEP 1 — DATASET PREPARATION")
print("=" * 60)

print("Dataset:", DATASET_DIR)

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_DIR}"
    )


# ============================================================
# COLLECT IMAGES
# ============================================================

records = []

for class_folder in sorted(DATASET_DIR.iterdir()):

    if not class_folder.is_dir():
        continue

    folder_name = class_folder.name.upper()

    if folder_name not in CLASS_MAPPING:
        print(
            f"Skipping unknown folder: {class_folder.name}"
        )
        continue

    class_name = CLASS_MAPPING[folder_name]

    for image_path in class_folder.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        records.append(
            {
                "File Path": str(image_path.resolve()),
                "Class": class_name,
            }
        )


master_df = pd.DataFrame(records)

if master_df.empty:
    raise RuntimeError("No images found.")


master_df = master_df.sort_values(
    "File Path"
).reset_index(drop=True)


# ============================================================
# VERIFY DATASET
# ============================================================

print("\nTotal images:", len(master_df))

print("\nClass distribution:")
print(master_df["Class"].value_counts().sort_index())

expected_classes = {
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TB",
}

if set(master_df["Class"].unique()) != expected_classes:
    raise ValueError(
        "Unexpected class names detected."
    )


# ============================================================
# HASH IMAGES
# ============================================================

def calculate_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


print("\nCalculating image hashes...")

master_df["File Hash"] = master_df["File Path"].apply(
    calculate_hash
)


# ============================================================
# FIND DUPLICATES
# ============================================================

duplicate_records = []

for file_hash, group in master_df.groupby("File Hash"):

    if len(group) <= 1:
        continue

    group = group.sort_values("File Path")

    original = group.iloc[0]["File Path"]

    for _, row in group.iloc[1:].iterrows():

        duplicate_records.append(
            {
                "Original": original,
                "Duplicate": row["File Path"],
                "Hash": file_hash,
            }
        )


duplicates_df = pd.DataFrame(
    duplicate_records,
    columns=[
        "Original",
        "Duplicate",
        "Hash",
    ],
)


print(
    "Duplicate records:",
    len(duplicates_df)
)


# ============================================================
# CREATE DUPLICATE GROUPS
# ============================================================

parent = {
    path: path
    for path in master_df["File Path"]
}


def find(x):

    if parent[x] != x:
        parent[x] = find(parent[x])

    return parent[x]


def union(x, y):

    root_x = find(x)
    root_y = find(y)

    if root_x != root_y:
        parent[root_y] = root_x


for _, row in duplicates_df.iterrows():

    union(
        row["Duplicate"],
        row["Original"]
    )


group_mapping = {}

next_group_id = 0

for path in master_df["File Path"]:

    root = find(path)

    if root not in group_mapping:

        group_mapping[root] = next_group_id
        next_group_id += 1

    parent[path] = group_mapping[root]


master_df["Group ID"] = (
    master_df["File Path"].map(parent)
)


print(
    "Unique groups:",
    master_df["Group ID"].nunique()
)


# ============================================================
# CHECK DUPLICATE GROUP CLASS CONSISTENCY
# ============================================================

group_class_counts = (
    master_df
    .groupby("Group ID")["Class"]
    .nunique()
)

mixed_groups = (
    group_class_counts > 1
).sum()

print(
    "Groups containing multiple classes:",
    mixed_groups
)

if mixed_groups > 0:
    raise RuntimeError(
        "A duplicate group contains multiple classes."
    )


# ============================================================
# CREATE GROUP-LEVEL DATA
# ============================================================

group_data = (
    master_df
    .groupby("Group ID")
    .agg(
        Class=("Class", "first"),
        Group_Size=("File Path", "count"),
    )
    .reset_index()
)


# ============================================================
# 70% TRAIN / 30% TEMP
# ============================================================

train_groups, temp_groups = train_test_split(
    group_data,
    test_size=0.30,
    stratify=group_data["Class"],
    random_state=RANDOM_STATE,
)


# ============================================================
# 15% VALIDATION / 15% TEST
# ============================================================

validation_groups, test_groups = train_test_split(
    temp_groups,
    test_size=0.50,
    stratify=temp_groups["Class"],
    random_state=RANDOM_STATE,
)


# ============================================================
# GROUP IDS
# ============================================================

train_ids = set(train_groups["Group ID"])
validation_ids = set(validation_groups["Group ID"])
test_ids = set(test_groups["Group ID"])


# ============================================================
# ASSIGN SPLITS
# ============================================================

def assign_split(group_id):

    if group_id in train_ids:
        return "train"

    if group_id in validation_ids:
        return "validation"

    if group_id in test_ids:
        return "test"

    raise RuntimeError(
        f"Unassigned group: {group_id}"
    )


master_df["Split"] = (
    master_df["Group ID"]
    .apply(assign_split)
)


# ============================================================
# VERIFY LEAKAGE
# ============================================================

group_split_counts = (
    master_df
    .groupby("Group ID")["Split"]
    .nunique()
)

leaking_groups = (
    group_split_counts > 1
).sum()


print("\n" + "=" * 40)
print("FINAL DATA SPLIT")
print("=" * 40)

print("\nImage counts:")
print(
    master_df["Split"]
    .value_counts()
    .sort_index()
)

print("\nClass distribution by split:")
print(
    pd.crosstab(
        master_df["Class"],
        master_df["Split"]
    )
)

print(
    "\nGroups appearing in multiple splits:",
    leaking_groups
)

if leaking_groups != 0:
    raise RuntimeError(
        "DATA LEAKAGE DETECTED."
    )


# ============================================================
# SAVE FILES
# ============================================================

master_df.to_csv(
    MASTER_PATH,
    index=False
)

duplicates_df.to_csv(
    DUPLICATES_PATH,
    index=False
)

master_df.to_csv(
    SPLIT_PATH,
    index=False
)


print("\nFiles saved:")
print(MASTER_PATH)
print(DUPLICATES_PATH)
print(SPLIT_PATH)

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)
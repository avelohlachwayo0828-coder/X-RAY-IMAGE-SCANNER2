from database import create_database, save_scan, get_scans


# Create database

create_database()


# Add a test scan

save_scan(
    "test_xray.png",
    "Possible TB findings detected"
)


# Display scans

scans = get_scans()


for scan in scans:

    print(scan)
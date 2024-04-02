import os
import sys
import shutil
# Function to find all .ply files in a directory and get their relative paths
def find_ply_files(folder_path):
    ply_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".ply"):
                ply_path = os.path.relpath(os.path.join(root, file), folder_path)
                ply_files.append(ply_path)

    filtered_ply = [s for s in ply_files if "30000" in s]
    return filtered_ply

# Check if the starting folder is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python script_name.py <starting_folder>")
    sys.exit(1)

# Get the starting folder from command-line arguments
starting_folder = sys.argv[1]

# Check if the starting folder exists
if not os.path.isdir(starting_folder):
    print("Error: Starting folder does not exist.")
    sys.exit(1)

# Find all .ply files and get their relative paths
ply_files = find_ply_files(starting_folder)

destination_folder = "BRAM"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

for index, s in enumerate(ply_files):
    print(s, destination_folder)
    shutil.copy(os.path.join(starting_folder, s), os.path.join(destination_folder, os.path.join(starting_folder, s).replace("\\", "")))

# Print the relative paths of .ply files
for ply_file in ply_files:
    print(ply_file)

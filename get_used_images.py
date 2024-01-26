import os
import re
import shutil

def find_and_copy_images(source_path, destination_path):
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    for filename in os.listdir(source_path):
        if filename.endswith(".jpg"):
            match = re.match(r'(\d+)\.jpg', filename)
            if match:
                idx = int(match.group(1))
                if idx % 15 == 1:
                    source_file_path = os.path.join(source_path, filename)
                    destination_file_path = os.path.join(destination_path, filename)
                    shutil.copy2(source_file_path, destination_file_path)

# Replace 'path/to/source/images' and 'path/to/destination/folder' with the actual paths
source_image_directory = 'truck_big/images_4'
destination_folder = 'used_images'

find_and_copy_images(source_image_directory, destination_folder)

print(f"Images where idx % 15 == 1 have been copied to {destination_folder}.")

import os
import imageio
import numpy as np

for i in range(100):
    print(int(i*29.97))

exit()

def images_are_identical(image_path1, image_path2):
    try:
        img1 = imageio.imread(image_path1)
        img2 = imageio.imread(image_path2)

        # Compare images using numpy array equality
        return np.array_equal(img1, img2)
    except Exception as e:
        # Handle exceptions (e.g., if images cannot be read)
        print(f"Error comparing images: {e}")
        return False

def find_identical_images(folder1, folder2):
    identical_images = []

    for root1, _, files1 in os.walk(folder1):
        
        for file1 in files1:
            print(file1)
            path1 = os.path.join(root1, file1)
            
            for root2, _, files2 in os.walk(folder2):
                for file2 in files2:
                    path2 = os.path.join(root2, file2)
                    
                    if images_are_identical(path1, path2):
                        print("found")
                        identical_images.append((path1, path2))
                        break

    return identical_images


# Replace 'path/to/folder1' and 'path/to/folder2' with the actual paths to your folders
folder1 = 'truck_big/images_4'
folder2 = 'tandt_db/tandt/truck/images'

identical_images_list = find_identical_images(folder1, folder2)

if identical_images_list:
    print("Identical images found:")
    for img1, img2 in identical_images_list:
        print(f"{img1} is identical to {img2}")
else:
    print("No identical images found.")
import os
import shutil

def remove_gt_subfolders(root_folder):

    for dirpath, dirnames, filenames in os.walk(root_folder):
        if 'gt' in dirnames:
            gt_path = os.path.join(dirpath, 'gt')
            print(f"Removing folder: {gt_path}")
            shutil.rmtree(gt_path)
            # If you want to prevent further os.walk from exploring this path, remove it from dirnames
            dirnames.remove('gt')

# Example usage
root_folder = 'output/datasets'
remove_gt_subfolders(root_folder)
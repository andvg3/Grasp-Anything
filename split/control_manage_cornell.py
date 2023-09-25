import os
import glob
import shutil
import pickle

def copy_files_with_structure(source_dir, destination_dir, filter_func=None):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for root, _, files in os.walk(source_dir):
        # Get the relative path of the current directory within the source directory
        relative_path = os.path.relpath(root, source_dir)
        # Create the corresponding directory structure in the destination directory
        dest_dir = os.path.join(destination_dir, relative_path)

        # Create the directory in the destination directory
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Copy each file in the current directory to the corresponding directory in the destination directory
        for file in files:
            source_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_dir, file)

            # Use the filter function if provided to check if the file should be copied
            if filter_func is None or filter_func(source_file_path):
                shutil.copy2(source_file_path, dest_file_path)  # Use shutil.copy2 to preserve metadata

# Example usage:
def seen_filter_cornell(file_path):
    idx = file_path.split('/')[-1].split('.')[0]
    idx = idx[:7]
    return idx in seen_filters

def unseen_filter_cornell(file_path):
    idx = file_path.split('/')[-1].split('.')[0]
    idx = idx[:7]
    return idx in unseen_filters

cornell_path = 'data/cornell'
cornell_filter_path = 'split/cornell'
source_directory = "data/cornell"
seen_destination_directory = "data/cornell_seen"
unseen_destination_directory = "data/cornell_unseen"

with open(os.path.join(cornell_filter_path, 'seen.obj'), 'rb') as f:
    seen_filters = pickle.load(f)

with open(os.path.join(cornell_filter_path, 'unseen.obj'), 'rb') as f:
    unseen_filters = pickle.load(f)

copy_files_with_structure(source_directory, seen_destination_directory, filter_func=seen_filter_cornell)
copy_files_with_structure(source_directory, unseen_destination_directory, filter_func=unseen_filter_cornell)

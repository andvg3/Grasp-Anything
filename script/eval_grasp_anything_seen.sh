#!/bin/bash

folder_path="$1"  # Replace '/path/to/your/folder' with the actual folder path

if [ ! -d "$folder_path" ]; then
    echo "Folder $folder_path not found."
    exit 1
fi

pattern="epoch_*"
for file in "$folder_path"/$pattern; do
    if [ -f "$file" ]; then
        echo "Running command with file: $file"
        python evaluate.py --dataset grasp-anything --dataset-path data/grasp-anything/ --iou-eval --use-depth 0 --seen 1 --split 0.99 --network "$file"  # Execute the command with the file as a parameter
    fi
done

echo "All files processed."

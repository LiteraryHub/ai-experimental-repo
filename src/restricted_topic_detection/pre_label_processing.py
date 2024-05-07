"""
A script for renaming and copying files from one directory to another with sequential numbering.

This script takes all files from a specified source directory, renames them sequentially starting from 1, maintaining their original file extensions, and copies the renamed files to a specified target directory. It is particularly useful for organizing files in a uniform naming scheme before processing them further.

Functions:
    rename_files(processed_dir, renamed_dir): Renames and copies files from the source directory to the target directory with sequential numbering.

Process Flow:
    1. The function `rename_files` is called with the source directory (`processed_dir`) and the target directory (`renamed_dir`) as arguments.
    2. It creates the target directory if it does not already exist, ensuring no error is thrown if the directory is already present.
    3. The script iterates over each file in the source directory, extracting the file extension and assigning a new file name based on its order in the list.
    4. Each file is then copied to the target directory with its new name, preserving the original file extension.
    5. The target directory path is returned upon successful completion of the renaming and copying process.

Main Execution:
    - Defines the source (`processed_dir`) and target (`renamed_dir`) directory paths.
    - Calls the `rename_files` function to perform the renaming and copying operation.
    - Prints the path to the target directory after the operation completes, indicating where the renamed files are located.

Usage:
    This script is intended to be run directly. Users must define the source and target directory paths before execution. It's ideal for data preprocessing stages, where files need to be organized into a consistent naming format for easier handling.

Note:
    - The script does not delete the original files in the source directory; it only copies them to the target directory with new names.
    - File numbering starts at 1 and increases sequentially for each file in the source directory, regardless of the original file names or types.
    - If the target directory already contains files, the script will add the newly named files without altering or deleting any existing files in the target directory.
"""
import os
import shutil
import re

import os
import shutil

def rename_files(processed_dir, renamed_dir):
    """
    Renames files in the specified directory and copies them to a new directory with a numbered prefix.

    Args:
        processed_dir (str): The path to the directory containing the files to be renamed.
        renamed_dir (str): The path to the directory where the renamed files will be copied.

    Returns:
        str: The path to the directory where the renamed files were copied.
    """
    # Get a list of all files in the directory
    files = os.listdir(processed_dir)
    
    os.makedirs(renamed_dir, exist_ok=True)
    
    # Iterate over the files and rename them
    for i, file in enumerate(files, start=1):
        # Get the file extension
        file_name, file_extension = os.path.splitext(file)
        
        # Rename the file with a number
        new_file_name = f"{i}{file_extension}"
        
        # Copy the file to the new directory with the new name
        shutil.copyfile(os.path.join(processed_dir, file), os.path.join(renamed_dir, new_file_name))
    
    return renamed_dir

if __name__ == "__main__":
    processed_dir = './src/restricted_topic_detection/processed_dataset'
    renamed_dir = './src/restricted_topic_detection/renamed_dataset'
    
    renamed_dir = rename_files(processed_dir, renamed_dir)
    print(f"Files renamed and copied to directory: {renamed_dir}")
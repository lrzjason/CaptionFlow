import os

def delete_old_txt_files(input_dir):
    # Walk through all files and directories in the given directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Check if the file is a .txt file and contains '_old' in its name
            if file.endswith('.txt') and '_old' in file:
                file_path = os.path.join(root, file)
                try:
                    # Delete the file
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

# input_dir = r'F:/ImageSet/kolors_anime_only_2k/florence' 
input_dir = r'F:/ImageSet/kolors_anime_only_2k/keta'
delete_old_txt_files(input_dir)
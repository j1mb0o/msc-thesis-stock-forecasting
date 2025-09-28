import glob
import os

def replace_in_file(file_path, old_str, new_str):
    """Replaces all occurrences of old_str with new_str in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        if old_str in content:
            new_content = content.replace(old_str, new_str)
            
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"Replaced content in: {file_path}")
        else:
            print(f"String '{old_str}' not found in: {file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def main():
    # Using glob to find all TimesFM config files
    config_files = glob.glob("configs/**/fm/**/*.yaml", recursive=True)
    
    old_path_str = "/root/"
    new_path_str = "/Users/dimitris/LU/Thesis/"
    
    for file_path in config_files:
        replace_in_file(file_path, old_path_str, new_path_str)

if __name__ == "__main__":
    main()

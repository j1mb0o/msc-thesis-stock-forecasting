import os
import glob

# Your replacements mapping
mapping = {
    'chronos_base': 'Chronos',
    'fm': 'TimesFM',
    'naive': 'Naive',
    'arima': 'ARIMA',
    'sundial': 'Sundial',
    '\\multirow': '\\midrule\n\\multirow',

    # Add more pairs
}

dir_path = '/Users/dimitris/LU/Thesis/Thesis-Master-Repo/thesis-code-new/tables/1d/train-less-year-log/*.tex'  # Your table files

for filepath in glob.glob(dir_path):
    backup = filepath + '.bak'
    # print(filepath)
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Apply all replacements
    for old, new in mapping.items():
        content = content.replace(old, new)
    
    # Write back (backup first)
    os.rename(filepath, backup)
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Updated {os.path.basename(filepath)}")

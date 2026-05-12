import os
import glob

mapping = {
    'chronos_base': 'Chronos',
    'fm': 'TimesFM',
    'naive': 'Naive',
    'arima': 'ARIMA',
    'sundial': 'Sundial',
    '\\multirow': '\\midrule\n\\multirow',
}

base_dir = '/Users/dimitris/LU/Thesis/Thesis-Master-Repo/thesis-code-new/tables/'

for timefreq in ['1d', '1h']:
    timefreq_path = os.path.join(base_dir, timefreq)
    for subfolder in os.listdir(timefreq_path):
        subfolder_path = os.path.join(timefreq_path, subfolder)
        if os.path.isdir(subfolder_path):
            for filepath in glob.glob(os.path.join(subfolder_path, '*.tex')):
                backup = filepath + '.bak'
                with open(filepath, 'r') as f:
                    content = f.read()

                for old, new in mapping.items():
                    content = content.replace(old, new)

                os.rename(filepath, backup)
                with open(filepath, 'w') as f:
                    f.write(content)

                print(f"Updated {os.path.basename(filepath)}")

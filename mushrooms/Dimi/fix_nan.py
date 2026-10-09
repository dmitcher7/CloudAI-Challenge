import json
import os

nb_path = r'C:\Users\dimit\Documents\01 - Thomas More\Thomas More Jaar 2\01 - Semester 1\Data Science\lessen\3 Jaar - Deep Learning\CloudAI_Project_Team\CloudAI-Challenge\mushrooms\Dimi\01_Dimi_Mushrooms_EDA_Model.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and "df_clean['stem-volume']" in "".join(cell['source']):
        source = "".join(cell['source'])
        # Remove the duplicate overwrite lines
        bad_lines = [
            "numeric_cols = ['cap-diameter', 'stem-height', 'stem-width']\n",
            "categorical_cols = df_clean.select_dtypes(include=['object']).columns.drop('class')\n"
        ]
        
        # Split source into lines, find the second occurrence of numeric_cols definition and remove it.
        # Actually it's simpler to just do a string replace for the duplicate part:
        new_source = source.replace(
            "numeric_cols = ['cap-diameter', 'stem-height', 'stem-width']\ncategorical_cols = df_clean.select_dtypes(include=['object']).columns.drop('class')\n",
            ""
        )
        
        lines = [line + '\n' for line in new_source.split('\n')]
        lines[-1] = lines[-1].strip('\n')
        cell['source'] = lines

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

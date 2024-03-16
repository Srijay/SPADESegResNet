"""

Script to group similar tissue types. This script takes original tissue masks or tissue labels
as input and form a new set of tissue masks where similar tissue regions are grouped

"""


import os
import shutil
import glob
import random
from PIL import Image
import numpy as np

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def read_tsv(path):
    code_dict = {}
    tissue_ids = {}
    with open(path, 'r', newline='') as tsv_file:
        # Skip the header line
        next(tsv_file, None)
        # Iterate over each line in the TSV file
        for line in tsv_file:
            # Split each line into two parts based on the space character
            parts = line.strip().split()
            # Ensure that there are two parts in each line
            if len(parts) == 2:
                # Assign the first part as the key and the second part as the value in the dictionary
                key, value = parts
                code_dict[int(value)] = key  # Assuming the values are integers, adjust if necessary
                tissue_ids[key] = int(value)
    return code_dict, tissue_ids

    
codes_path = "./data/gtruth_codes.tsv"
code_dict, tissue_ids = read_tsv(codes_path)
input_dir = "./data/labels"
output_dir = "./data/grouped_labels"
tissue_groups = {1: ['tumor','angioinvasion', 'dcis'], 2: ['stroma'], 3: ['lymphocytic_infiltrate', 'plasma_cells', 'other_immune_infiltrate'], 4: ['necrosis_or_debris'], 5: ['others']}

def create_mapping_dict(tissue_groups):
    mapping_dict = {}

    for group, tissues in tissue_groups.items():
        for tissue in tissues:
            mapping_dict[tissue] = group

    return mapping_dict

tissue_mapping_dict = create_mapping_dict(tissue_groups)
tissue_mapping_dict.pop('others')

tissue_mapping_dict = {tissue_ids[key]: tissue_mapping_dict[key] for key in tissue_mapping_dict.keys()}

mkdir(output_dir)

image_files = [file for file in os.listdir(input_dir) if file.endswith(".png")]

for imname in image_files:
    image_path = os.path.join(input_dir, imname)
    output_path = os.path.join(output_dir, imname)
    mask = Image.open(image_path)
    mask = np.asarray(mask)
    for i in range(1,len(code_dict)):
        if i in tissue_mapping_dict:
            mask = np.where(mask == i, tissue_mapping_dict[i], mask)
        else:
            mask = np.where(mask == i, 5, mask) #others class
    Image.fromarray(mask).save(output_path)

print("Done")




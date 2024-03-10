from PIL import Image
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt

def read_image(img_path):
    img = Image.open(img_path)
    img = np.asarray(img)
    return img

def get_class_mapping(tissue_groups, tissue_list):
    result = [next((group for group, values in tissue_groups.items() if item in values), 5) for item in tissue_list]
    result = list(dict.fromkeys(result))
    return result

codes_path = r"F:\Datasets\BCSS_InstaDeep\meta\gtruth_codes.tsv"
code_dict = {}
tissue_ids = {}
# Open the TSV file in read mode with the appropriate newline parameter
with open(codes_path, 'r', newline='') as tsv_file:
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

labels_path = r'F:\Datasets\BCSS_InstaDeep\grouped_masks'
output_plot_name = 'tissue_barplot_grouping.png'
output_pixels_plot_name = 'tissue_pixels_barplot_grouping.png'
do_grouping = True
class_counts = {key: 0 for key in range(1, 22)} # tissue types with number of images having those tissues
class_pixel_props = {key: 0 for key in range(1, 22)} # tissue types with total pixel proportions
tissue_groups = {1: ['tumor','angioinvasion', 'dcis'], 2: ['stroma'], 3: ['lymphocytic_infiltrate', 'plasma_cells', 'other_immune_infiltrate'], 4: ['necrosis_or_debris'], 5: ['others']}
if(do_grouping):
    class_counts = {key: 0 for key in range(1, len(tissue_groups)+1)}
    class_pixel_props = {key: 0 for key in range(1, len(tissue_groups)+1)}

for label_file in os.listdir(labels_path):
    label_path = os.path.join(labels_path, label_file)
    labels = read_image(label_path)
    classes_present = list(np.unique(labels))
    if 0 in classes_present:
        classes_present.remove(0)
    tissues = [code_dict[index] for index in classes_present]
    if(do_grouping):
        classes_present = get_class_mapping(tissue_groups, tissues)
    for key in classes_present:
        if key in class_counts:
            class_counts[key] += 1
        class_pixel_props[key]+=np.count_nonzero(labels == key)


print("Labels counts: ", class_counts)

# Extract keys and counts from the dictionary
keys = list(class_counts.keys())
class_labels = [code_dict[key] for key in keys]
if(do_grouping):
    class_labels = ['tumor', 'stroma', 'inflammatory', 'necrosis', 'others']
counts = list(class_counts.values())
pixels_counts = list(class_pixel_props.values())
pixels_counts = [(value / sum(pixels_counts)) * 100 for value in pixels_counts]
class_weights = [x/100 for x in pixels_counts]
class_weights = [1-x for x in class_weights]
print('percentages: ',pixels_counts)
print('class weights: ',class_weights)

plt.figure(figsize=(10, 6))
bar_width = 0.15
plt.bar(class_labels, counts, color='blue', width=bar_width)
bar_positions = range(len(class_labels))
plt.xticks(rotation=45, ha='right')
plt.xlabel('Tissue types', fontsize=17)
plt.ylabel('Counts', fontsize=17)
plt.title('Distribution of number of images with tissue types (with grouping)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.show()
plt.savefig(output_plot_name)

plt.figure(figsize=(10, 6))
bar_width = 0.15
plt.bar(class_labels, pixels_counts, color='blue', width=bar_width)
bar_positions = range(len(class_labels))
plt.xticks(rotation=45, ha='right')
plt.xlabel('Tissue types', fontsize=17)
plt.ylabel('Pixels Percentage', fontsize=17)
plt.title('Pixels wise distribution (with grouping)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.show()
plt.savefig(output_pixels_plot_name)
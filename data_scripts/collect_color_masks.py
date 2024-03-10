import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import glob
import csv
import PIL
from PIL import Image
import pandas as pd
import colorsys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PIL.Image.MAX_IMAGE_PIXELS = 933120000

masks_folder = r"F:\Datasets\BCSS_InstaDeep\splits\train\grouped_masks"
outfolder = r"F:\Datasets\BCSS_InstaDeep\splits\train\grouped_color_masks"
masks_paths = glob.glob(os.path.join(masks_folder,"*.png"))

codes_path = r"F:\Datasets\BCSS_InstaDeep\meta\gtruth_codes.tsv"
code_dict = {}
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
            code_dict[key] = int(value)  # Assuming the values are integers, adjust if necessary
code_dict_reversed = {value: key for key, value in code_dict.items()}

if('grouped_masks' in masks_folder):
    code_dict_reversed = {0: 'outside roi', 1: 'tumor', 2: 'stroma', 3: 'inflammatory', 4: 'necrosis', 5: 'others'}

code_len = len(code_dict)

if not os.path.exists(outfolder):
        os.makedirs(outfolder)

def generate_colors(n):
  """
  Generate random colors.
  To get visually distinct colors, generate them in HSV space then
  convert to RGB.
  """
  brightness = 0.7
  hsv = [(i / n, 1, brightness) for i in range(n)]
  colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
  colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
  return colors

def extract_image(mask_path):
    mask_name = os.path.split(mask_path)[1]
    mask_name = 'TCGA-BH-A0WA-DX1_xmin56581_ymin24774_MPP.png'
    mask_path = os.path.join(masks_folder, mask_name)
    print(mask_name)
    mask_name = mask_name.split(".")[0]+".png"
    mask = Image.open(mask_path)
    mask_np = np.asarray(mask)
    print(mask_np)
    print(mask_np.shape)
    #plot legend
    present_classes_ids = list(np.unique(mask_np))
    print(len(present_classes_ids))
    print(present_classes_ids)
    print(code_dict_reversed)
    present_classes = [code_dict_reversed[id] for id in present_classes_ids]
    present_classes_colors = [colors[i] for i in present_classes_ids]
    present_classes_colors = [(r / 255, g / 255, b / 255) for r, g, b in present_classes_colors]

    legend_elements = [Line2D([0], [0], marker='o', color=color, label=f"{string}")
                       for id, string, color in zip(present_classes_ids, present_classes, present_classes_colors)]
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.subplots_adjust(left=0.2)

    # ax.legend(handles=legend_elements, loc='best', bbox_to_anchor=(1, 1), prop={'size': 18})
    ax.legend(handles=legend_elements, loc='best', prop={'size': 18},
              ncol=len(legend_elements))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()
    exit()

    w, h = mask_np.shape
    new_mk = np.empty([w, h, 3])
    for i in range(0,w):
        for j in range(0,h):
            new_mk[i][j] = colors[mask_np[i][j]]
    new_mk = new_mk / 255.0
    matplotlib.image.imsave(os.path.join(outfolder,mask_name), new_mk)
    print(mask_name)

# colors = generate_colors(code_len)
colors = generate_colors(6)


for path in masks_paths:
    extract_image(path)
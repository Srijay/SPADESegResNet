from PIL import Image
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt

def read_image(img_path):
    img = Image.open(img_path)
    img = np.asarray(img)
    return img

labels_path = r'F:\Datasets\BCSS_InstaDeep\masks'
image_sizes_dict = {}

for label_file in os.listdir(labels_path):
    label_path = os.path.join(labels_path, label_file)
    labels = read_image(label_path)
    image_sizes_dict[labels.shape[0]*labels.shape[1]] = labels.shape

min_key = min(image_sizes_dict, key=image_sizes_dict.get)
max_key = max(image_sizes_dict, key=image_sizes_dict.get)

print('Max image size: ',image_sizes_dict[max_key])
print('Min image size: ',image_sizes_dict[min_key])
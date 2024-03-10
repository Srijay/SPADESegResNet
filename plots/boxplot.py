"""

Script to construct boxplots

"""

import matplotlib.pyplot as plt
import json

spaderesnet_dice_path = r'F:\Datasets\BCSS_InstaDeep\splits\test\cropped\768\results\spaderesnet\spaderesnet_dice.json'
unet_dice_path = r'F:\Datasets\BCSS_InstaDeep\splits\test\cropped\768\results\spaderesnet\spaderesnet_dice.json'
unetpp_dice_path = r'F:\Datasets\BCSS_InstaDeep\splits\test\cropped\768\results\spaderesnet\spaderesnet_dice.json'

metric = 'Dice score'

tissue_id_dict = {1: 'Tumor', 2: 'Stroma', 3: 'Inflammatory', 4: 'Necrosis', 5: 'Other'}

with open(spaderesnet_dice_path, 'r') as json_file:
    spaderesnet_loaded_dict = json.load(json_file)

with open(unet_dice_path, 'r') as json_file:
    unet_loaded_dict = json.load(json_file)

with open(unetpp_dice_path, 'r') as json_file:
    unetpp_loaded_dict = json.load(json_file)

labels = ['SPADE-ResNet', 'UNet', 'UNet++']
colors = ['lightblue', 'lightcoral', 'lightgreen']

for label in range(1, 6):
    boxplot = plt.boxplot([spaderesnet_loaded_dict[str(label)], unet_loaded_dict[str(label)], unetpp_loaded_dict[str(label)]], labels=labels, vert=True, patch_artist=True, medianprops=dict(color='black'))
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    plt.xlabel('Methods')
    plt.ylabel(metric)
    plt.title(tissue_id_dict[label])
    plt.savefig(metric+'_'+tissue_id_dict[label]+'.png')
    plt.clf()
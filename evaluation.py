"""

Script to construct boxplots, computation of mean, standard deviation and p-values

"""

import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.stats import mannwhitneyu

spaderesnet_dice_path = './output/spaderesnet_dice.json'
unet_dice_path = './output/spaderesnet_auc.json/unet_dice.json'
unetpp_dice_path = './output/spaderesnet_auc.json/unet++_dice.json'

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
    print('Mean and Std for SPADESegResNet, ',tissue_id_dict[label], ' is ',np.mean(spaderesnet_loaded_dict[str(label)]), ' std: ', np.std(spaderesnet_loaded_dict[str(label)]))
    print('Mean and Std for UNet, ',tissue_id_dict[label], ' is ',np.mean(unet_loaded_dict[str(label)]), ' std: ', np.std(unet_loaded_dict[str(label)]))
    print('Mean and Std for UNet++, ',tissue_id_dict[label], ' is ',np.mean(unetpp_loaded_dict[str(label)]), ' std: ', np.std(unetpp_loaded_dict[str(label)]))
    U1, p_unet = mannwhitneyu(spaderesnet_loaded_dict[str(label)], unet_loaded_dict[str(label)], method="auto")
    U1, p_unetpp = mannwhitneyu(spaderesnet_loaded_dict[str(label)], unetpp_loaded_dict[str(label)], method="auto")
    print("P value for SPADESegResNet vs UNet: ",p_unet)
    print("P value for SPADESegResNet vs UNet++: ",p_unetpp)
    boxplot = plt.boxplot([spaderesnet_loaded_dict[str(label)], unet_loaded_dict[str(label)], unetpp_loaded_dict[str(label)]], labels=labels, vert=True, patch_artist=True, medianprops=dict(color='black'))
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.show()
    plt.savefig(metric+'_'+tissue_id_dict[label]+'.png')
    plt.clf()

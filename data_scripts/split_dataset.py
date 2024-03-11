"""

Script to split the dataset into training and testing set

"""

import os
import shutil
import glob
import random

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

input_dir = "./data"
output_dir = "./data/splits/"
split_factor = 0.7

mask_dir = os.path.join(input_dir,"labels")
image_dir = os.path.join(input_dir,"images")

output_train_dir = os.path.join(output_dir,"train")
output_test_dir = os.path.join(output_dir,"test")

train_mask_dir = os.path.join(output_train_dir,"labels")
train_image_dir = os.path.join(output_train_dir,"images")
test_mask_dir = os.path.join(output_test_dir,"labels")
test_image_dir = os.path.join(output_test_dir,"images")

mkdir(train_mask_dir)
mkdir(train_image_dir)
mkdir(test_mask_dir)
mkdir(test_image_dir)

mask_paths = glob.glob(os.path.join(mask_dir,"*.png"))

imnames = [os.path.split(path)[1].split(".")[0] for path in mask_paths]

train_l = int(split_factor*len(imnames))
test_l = len(imnames) - train_l

train_imnames = random.sample(imnames,train_l)

print("train length ",train_l)
print("test length ",test_l)

for imname in imnames:
    imname_start = '-'.join(imname.split('-')[:-1])
    image_name = imname_start + '-0.2500'

    if(imname in train_imnames):
        shutil.copy(os.path.join(mask_dir,image_name+'.png'), os.path.join(train_mask_dir,imname_start+'.png'))
        shutil.copy(os.path.join(image_dir,image_name+'.png'), os.path.join(train_image_dir,imname_start+'.png'))
    else:
        shutil.copy(os.path.join(mask_dir, image_name + '.png'), os.path.join(test_mask_dir,imname_start+'.png'))
        shutil.copy(os.path.join(image_dir, image_name + '.png'), os.path.join(test_image_dir,imname_start+'.png'))

print("Done")




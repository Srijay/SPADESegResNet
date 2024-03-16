"""

Script to extract image tiles from the BCSS dataset

"""

import glob
import os
from PIL import Image
import numpy as np
import PIL

folder_path = "./data/splits/train"
masks_input_folder = os.path.join(folder_path, "grouped_labels")
images_input_folder = os.path.join(folder_path, "images")

output_dir = "./data/splits/train/cropped/768"
masks_output_folder = os.path.join(output_dir, "grouped_labels")
images_output_folder = os.path.join(output_dir, "images")

if not os.path.exists(masks_output_folder):
        os.makedirs(masks_output_folder)
if not os.path.exists(images_output_folder):
        os.makedirs(images_output_folder)

patchsize = 768
pad = 0
stride = 512
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def CropImage(imgname):

    masks_image_path = os.path.join(masks_input_folder,imgname+".png")
    images_image_path = os.path.join(images_input_folder, imgname+".png")
    image_initial = imgname

    mask_im_ = Image.open(masks_image_path)
    image_im_ = Image.open(images_image_path)

    width, height = image_im_.size
    if (pad):
        new_size = (width + pad, height + pad)
        image_im  = Image.new("RGB", new_size)
        image_im.paste(image_im_, ((new_size[0] - width) // 2,
                          (new_size[1] - height) // 2))
        mask_im_ = np.asarray(mask_im_)
        mask_im_ = np.pad(mask_im_, ((int(pad/2), int(pad/2)), (int(pad/2), int(pad/2))), mode='constant', constant_values=0)
        mask_im = Image.fromarray(mask_im_)
    else:
        image_im, mask_im = image_im_, mask_im_

    x = 0
    y = 0
    right = 0
    bottom = 0

    while (bottom < height):
        while (right < width):
            left = x
            top = y
            right = left + patchsize
            bottom = top + patchsize
            if (right > width):
                offset = right - width
                right -= offset
                left -= offset
            if (bottom > height):
                offset = bottom - height
                bottom -= offset
                top -= offset

            im_crop_name = image_initial + "_" + str(left) + "_" + str(top) + ".png"

            im_crop_mask = mask_im.crop((left, top, right, bottom))

            im_crop_image = image_im.crop((left, top, right, bottom))

            im_crop_mask_np = np.asarray(im_crop_mask)

            dontcare_percentage = (np.count_nonzero(im_crop_mask_np == 0) / im_crop_mask_np.size) * 100 # Ignore tissue tiles with don't care percentage greater than 50%

            if (dontcare_percentage <= 50):
                output_mask_path = os.path.join(masks_output_folder, im_crop_name)
                output_image_path = os.path.join(images_output_folder, im_crop_name)
                im_crop_mask.save(output_mask_path)
                im_crop_image.save(output_image_path)

            x += stride

        x = 0
        right = 0
        y += stride

avgs = []
masks_image_paths = glob.glob(os.path.join(masks_input_folder,"*.png"))

image_names = []
for path in masks_image_paths:
    imgname = os.path.split(path)[1].split('.')[0]
    CropImage(imgname)

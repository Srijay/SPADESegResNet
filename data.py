import os
import sys
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

sys.path.insert(0,os.getcwd())


class BCSSDataset(Dataset):

    def __init__(self, parameters, mode='train', seed=21):
        super(Dataset, self).__init__()

        self.images_dir = parameters[mode+'_images']
        self.masks_dir = parameters[mode+'_masks']
        self.image_ids = os.listdir(self.images_dir)

    def read_image(self,img_path):
        img = Image.open(img_path)
        img = np.asarray(img)
        if(np.max(img)>100): #Normalize the image pixel values between (0,1)
            img = img/255.0
        return img

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):

        image_id = self.image_ids[index]
        image_path = os.path.join(self.images_dir,image_id)
        label_path = os.path.join(self.masks_dir,image_id)

        transform = T.Compose([T.ToTensor()])
        image = self.read_image(image_path)
        image = transform(image)

        labels = self.read_image(label_path)

        return image_id, image, labels

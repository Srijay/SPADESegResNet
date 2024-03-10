"""

The main script

"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import colorsys
import configparser
import shutil
import json
import matplotlib

from data import BCSSDataset
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from early_stopping import EarlyStopper
from model.unet import UNet
from model.nestedunet import NestedUNet
from model.spadesegresnet import SPADEResNet
from metrics import *
from utils import *


#Global parameters
config = configparser.ConfigParser()
config.read('config.txt')
parameters = config['parameters']
output_dir = os.path.join(parameters['output_dir'], parameters['model_name'])
test_params = config['test']


def load_model(model_name=''):
    if(model_name=='spaderesnet'):
        model = SPADEResNet(input_nc=3, output_nc=6)
    elif(model_name=='unet'):
        model = UNet(input_nc=3, output_nc=6)
    elif(model_name=='unet++'):
        model = NestedUNet(input_nc=3, output_nc=6)
    else:
        print('model not supported: ',model_name)
        exit()
    model = nn.DataParallel(model)
    model = model.cuda()
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    print("Num trainable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if (int(parameters['restore_model'])):
        model_path = test_params['test_model_path']
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path), strict=True)
            print("model loaded")
        else:
            print("Give a proper path to model")
            exit()
    return model

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

def generate_colored_image(image_id, labels, dirname):
    colored_output_dir = os.path.join(output_dir, dirname)
    mkdir(colored_output_dir)
    colors = generate_colors(6)
    w, h = labels.shape
    new_mk = np.empty([w, h, 3])
    for i in range(0,w):
        for j in range(0,h):
            new_mk[i][j] = colors[labels[i][j]]
    new_mk = new_mk / 255.0
    matplotlib.image.imsave(os.path.join(colored_output_dir,image_id), new_mk)

def train(model, model_name):

    model.train()
    batch_size = int(parameters['batch_size'])

    bcss_data = BCSSDataset(parameters, mode='train')
    train_data, valid_data = torch.utils.data.random_split(bcss_data, [0.8,0.2], generator=torch.Generator().manual_seed(42))
    print("Length of training data: ",len(train_data))
    print("Length of validation data: ",len(valid_data))
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=float(parameters['lr']))
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    early_stopper = EarlyStopper(patience=20, min_delta=0)
    class_weights = torch.Tensor([0.01, 0.5998164516984599, 0.6570353649810721, 0.8685892941340623, 0.9282517140683683, 0.9463071751180375]).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    start_epoch = 0

    model_dir = os.path.join(parameters['model_dir'], parameters['model_name'])
    mkdir(model_dir)
    shutil.copyfile('config.txt', os.path.join(model_dir, 'config.txt'))

    print("Starting the Training")
    current_validation_loss = 1000000
    for epoch in range(start_epoch, int(parameters['epochs'])):
        epoch_loss = 0.0  # for one full epoch
        for (b_idx, batch) in enumerate(train_dataloader):
            if(b_idx%100==0):
                print("ITER: ",b_idx)
            image_id, image, gt_labels = batch
            image = image.cuda()
            gt_labels = gt_labels.long().cuda()
            optimizer.zero_grad()
            pred_labels = model(image.float())
            loss = criterion(pred_labels, gt_labels)
            epoch_loss += loss.item()  # accumulate
            loss.backward()  # compute gradients
            optimizer.step()  # update weights

        print("epoch number = %4d  |  loss = %0.4f" % (epoch, epoch_loss/batch_size))

        print("validating:")
        valid_epoch_loss = 0.0  # run validation for one full epoch
        with torch.no_grad():
            for (b_idx, batch) in enumerate(valid_dataloader):
                _, image, gt_labels = batch
                image = image.cuda()
                gt_labels = gt_labels.long().cuda()
                pred_labels = model(image.float())
                valid_loss = criterion(pred_labels, gt_labels)
                valid_epoch_loss += valid_loss.item()  # accumulate

        validation_loss = valid_epoch_loss/batch_size
        print("Validation loss = ",valid_epoch_loss/batch_size)

        scheduler.step(valid_epoch_loss)

        if early_stopper.early_stop(valid_epoch_loss):
            print('early stopping')
            break

        if(validation_loss < current_validation_loss):
            current_validation_loss = validation_loss
            print("Saving the model")
            torch.save(model.state_dict(), os.path.join(model_dir, model_name+".pt"))

    print("Model trained")

def test(model, mode='patch'):
    mkdir(output_dir)
    model.eval()
    test_data = BCSSDataset(parameters, mode='test')
    print("Length of testing data: ", len(test_data))
    dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    dice_scores_dict = {}
    auc_scores_dict = {}
    accuracy_scores_dict = {}
    for label in range(1, 6):
        dice_scores_dict[label]=[]
        auc_scores_dict[label]=[]
        accuracy_scores_dict[label]=[]
    with torch.no_grad():
        for (b_idx, batch) in enumerate(dataloader):
            image_id, image, gt_labels = batch
            image = image.cuda()
            if(mode=='patch'):
                pred_labels = model(image.float())
            else:
                patch_size = 768
                stride = 700
                generator_output_size = 768
                pred_labels = torch.zeros(1, 6, image.shape[2], image.shape[3]).cuda()
                counter_tensor = torch.zeros(1, 1, image.shape[2], image.shape[3]).cuda()
                for i in range(0, image.shape[2] - patch_size + 1, stride):
                    for j in range(0, image.shape[3] - patch_size + 1, stride):
                        i_lowered = min(i, image.shape[2] - patch_size)
                        j_lowered = min(j, image.shape[3] - patch_size)
                        patch = image[:, :, i_lowered:i_lowered + patch_size, j_lowered:j_lowered + patch_size]
                        pred_labels_patch = model(patch.float())
                        update_region_i = i_lowered + (patch_size - generator_output_size) // 2
                        update_region_j = j_lowered + (patch_size - generator_output_size) // 2
                        pred_labels[:, :, update_region_i:update_region_i + generator_output_size, update_region_j:update_region_j + generator_output_size] += pred_labels_patch
                        counter_tensor[:, :, update_region_i:update_region_i + generator_output_size, update_region_j:update_region_j + generator_output_size] += 1
                pred_labels /= counter_tensor

            pred_labels = np.argmax(pred_labels.cpu().numpy(), axis=1)
            pred_labels = pred_labels[0]
            gt_labels = gt_labels[0].cpu().numpy()
            generate_colored_image(image_id[0], pred_labels, 'pred')
            generate_colored_image(image_id[0], gt_labels, 'gt')
            for label in range(1,6):
                # Dice score computation
                dice_score = compute_dice(pred_labels, gt_labels, label)
                if(dice_score!=-1):
                    dice_scores_dict[label].append(dice_score)
                # AUC-ROC computation
                auc_roc = compute_auc_roc(pred_labels, gt_labels, label)
                if (auc_roc != -1):
                    auc_scores_dict[label].append(auc_roc)
                # Accuracy computation
                accuracy = compute_accuracy(pred_labels, gt_labels, label)
                if (accuracy != -1):
                    accuracy_scores_dict[label].append(accuracy)
    with open(os.path.join(output_dir, model_name+'_dice.json'), 'w') as json_file:
        json.dump(dice_scores_dict, json_file)
    with open(os.path.join(output_dir, model_name+'_auc.json'), 'w') as json_file:
        json.dump(auc_scores_dict, json_file)
    with open(os.path.join(output_dir, model_name+'_accuracy.json'), 'w') as json_file:
        json.dump(accuracy_scores_dict, json_file)


model_name = parameters['model_name']
if(parameters['mode'] == 'train'):
    model = load_model(model_name=model_name)
    train(model, model_name)
else:
    mode = parameters['mode'].split("_")[1]
    model = load_model(model_name=model_name)
    test(model, mode=mode)
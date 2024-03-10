"""

Script includes functions to compute evaluation metrics

"""

import numpy as np
from sklearn.metrics import roc_auc_score

def compute_dice(pred, gt, label):
    pred = pred.flatten()
    gt = gt.flatten()
    pred[pred!=label] = 0
    pred[pred==label] = 1
    gt[gt != label] = 0
    gt[gt == label] = 1
    pred_pixels = np.sum(pred)
    gt_pixels = np.sum(gt)
    denom = (pred_pixels + gt_pixels)
    if (gt_pixels == 0):
        return -1
    return np.sum(pred[gt == 1]) * 2.0 / denom

def compute_auc_roc(pred, gt, label):
    pred_binary = np.where(pred == label, 1, 0)
    gt_binary = np.where(gt == label, 1, 0)
    if (np.sum(gt_binary) == 0 or np.all(gt_binary == 1) or np.all(pred_binary == 1)):
        return -1
    if (np.sum(pred_binary) == 0):
        return 0
    pred_flat = pred_binary.flatten()
    gt_flat = gt_binary.flatten()
    auc_roc = roc_auc_score(gt_flat, pred_flat)
    return auc_roc

def compute_accuracy(pred, gt, label):
    if (np.sum(gt)==0):
        return -1
    pred_binary = (pred == label).astype(int)
    gt_binary = (gt == label).astype(int)
    correct_predictions = np.sum(pred_binary == gt_binary)
    total_predictions = pred.size
    accuracy = correct_predictions / total_predictions
    return accuracy

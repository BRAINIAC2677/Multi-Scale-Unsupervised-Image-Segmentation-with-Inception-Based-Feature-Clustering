#from __future__ import print_function
import cv2
import numpy as np
import torch.nn.functional as F

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'
]

VOC_PALETTE = {
    (0, 0, 0): 0, (128, 0, 0): 1, (0, 128, 0): 2, (128, 128, 0): 3, 
    (0, 0, 128): 4, (128, 0, 128): 5, (0, 128, 128): 6, (128, 128, 128): 7,
    (64, 0, 0): 8, (192, 0, 0): 9, (64, 128, 0): 10, (192, 128, 0): 11, 
    (64, 0, 128): 12, (192, 0, 128): 13, (64, 128, 128): 14, (192, 128, 128): 15,
    (0, 64, 0): 16, (128, 64, 0): 17, (0, 192, 0): 18, (128, 192, 0): 19, 
    (0, 64, 128): 20
}

def convert_voc_gt_to_labels(gt_path):
    gt_image = cv2.imread(gt_path)
    gt_array = np.array(gt_image)

    label_map = np.zeros(gt_array.shape[:2], dtype=np.uint8)

    for rgb, label in VOC_PALETTE.items():
        mask = np.all(gt_array == rgb, axis=-1)
        label_map[mask] = label

    return label_map

def calculate_miou(pred_labels, gt_labels):
    # Reshape the arrays into 1D
    gt_labels_1d = gt_labels.reshape(-1)
    pred_labels_1d = pred_labels.reshape(-1)
    
    # Get unique labels from ground truth
    unique_labels = np.unique(gt_labels_1d)
    
    miou_list = []
    
    # Iterate over each unique label
    for label in unique_labels:
        # Find indices where the current label is present in ground truth
        gt_indices = np.where(gt_labels_1d == label)[0]
        
        # Get the predicted labels at these indices
        pred_at_gt_indices = pred_labels_1d[gt_indices]
        
        # Get unique predicted labels at these indices
        unique_pred_labels = np.unique(pred_at_gt_indices)
        
        # Calculate histogram and fractions for IoU calculation
        hist = [np.sum(pred_at_gt_indices == u) for u in unique_pred_labels]
        fractions = [len(gt_indices) + np.sum(pred_labels_1d == u) - np.sum(pred_at_gt_indices == u) for u in unique_pred_labels]
        
        # Calculate IoU for each unique predicted label
        ious = np.array(hist) / np.array(fractions, dtype='float')
        
        # Append the maximum IoU for the current label to the list
        miou_list.append(np.max(ious))
    
    mean_iou = np.mean(miou_list)
    return mean_iou

def evaluate(pred_labels, gt_path):
    gt_labels = convert_voc_gt_to_labels(gt_path) 
    return calculate_miou(pred_labels, gt_labels)
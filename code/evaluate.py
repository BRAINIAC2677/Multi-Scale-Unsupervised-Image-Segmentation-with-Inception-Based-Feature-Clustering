#from __future__ import print_function
import cv2
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score


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


def convert_gt_to_labels(gt_path):
    gt_image = cv2.imread(gt_path)
    gt_array = np.array(gt_image)

    label_map = np.zeros(gt_array.shape[:2], dtype=np.uint8)

    # now assign labels starting from 0 to the colors
    for i, rgb in enumerate(np.unique(gt_array.reshape(-1, gt_array.shape[2]), axis=0)):
        mask = np.all(gt_array == rgb, axis=-1)
        label_map[mask] = i
    
    return label_map


def remap_labels(pred_labels, gt_labels):
    """
    Remaps predicted labels to match the ground truth labels using the Hungarian algorithm.
    
    Args:
        pred_labels (np.ndarray): Predicted segmentation labels (2D or 3D array).
        gt_labels (np.ndarray): Ground truth segmentation labels (2D or 3D array).
        
    Returns:
        np.ndarray: Remapped predicted labels.
    """
    # Reshape the arrays into 1D
    gt_labels_1d = gt_labels.reshape(-1)
    pred_labels_1d = pred_labels.reshape(-1)
    
    # Get unique ground truth and predicted labels
    unique_gt_labels = np.unique(gt_labels_1d)
    unique_pred_labels = np.unique(pred_labels_1d)
    
    # Initialize the cost matrix (rows: ground truth, columns: predicted)
    cost_matrix = np.zeros((len(unique_gt_labels), len(unique_pred_labels)))
    
    # Populate the cost matrix (negative intersection for Hungarian minimization)
    for i, gt_label in enumerate(unique_gt_labels):
        for j, pred_label in enumerate(unique_pred_labels):
            # Find the intersection (common pixels) between the labels
            gt_indices = np.where(gt_labels_1d == gt_label)[0]
            pred_indices = np.where(pred_labels_1d == pred_label)[0]
            intersection = len(np.intersect1d(gt_indices, pred_indices))
            
            # Use -intersection as cost (minimizing negative intersection maximizes accuracy)
            cost_matrix[i, j] = -intersection
    
    # Solve the assignment problem using the Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Create a mapping from predicted labels to ground truth labels
    pred_to_gt_mapping = {}
    for row, col in zip(row_indices, col_indices):
        pred_to_gt_mapping[unique_pred_labels[col]] = unique_gt_labels[row]
    
    # Relabel the predicted labels based on the mapping
    remapped_pred_labels = np.zeros_like(pred_labels_1d)
    for pred_label, gt_label in pred_to_gt_mapping.items():
        remapped_pred_labels[pred_labels_1d == pred_label] = gt_label
    
    # Reshape the remapped labels to the original shape
    remapped_pred_labels = remapped_pred_labels.reshape(pred_labels.shape)
    return remapped_pred_labels


def calculate_miou(pred_labels, gt_labels):
    """
    Calculates the mIoU using the Hungarian matching algorithm.
    
    Args:
        pred_labels (np.ndarray): Predicted segmentation labels (2D or 3D array).
        gt_labels (np.ndarray): Ground truth segmentation labels (2D or 3D array).
        
    Returns:
        float: Mean Intersection over Union (mIoU).
    """
    # Reshape the arrays into 1D
    gt_labels_1d = gt_labels.reshape(-1)
    pred_labels_1d = pred_labels.reshape(-1)
    
    # Get unique ground truth and predicted labels
    unique_gt_labels = np.unique(gt_labels_1d)
    unique_pred_labels = np.unique(pred_labels_1d)
    
    # Initialize IoU cost matrix (rows: ground truth, columns: predicted)
    cost_matrix = np.zeros((len(unique_gt_labels), len(unique_pred_labels)))
    
    # Compute IoU for each ground truth and predicted label pair
    for i, gt_label in enumerate(unique_gt_labels):
        for j, pred_label in enumerate(unique_pred_labels):
            # Find indices for the current ground truth and predicted label
            gt_indices = np.where(gt_labels_1d == gt_label)[0]
            pred_indices = np.where(pred_labels_1d == pred_label)[0]
            
            # Calculate intersection and union
            intersection = len(np.intersect1d(gt_indices, pred_indices))
            union = len(np.union1d(gt_indices, pred_indices))
            
            # IoU = intersection / union, set cost as 1 - IoU
            cost_matrix[i, j] = 1 - (intersection / union if union > 0 else 0)
    
    # Apply Hungarian algorithm to find optimal matching
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Calculate mIoU using the optimal matching
    matched_ious = []
    for row, col in zip(row_indices, col_indices):
        iou = 1 - cost_matrix[row, col]  # Recover IoU from the cost matrix
        matched_ious.append(iou)
    
    # Compute the mean IoU over all matched pairs
    mean_iou = np.mean(matched_ious)
    return mean_iou


def calculate_accuracy(pred_labels, gt_labels):
    """
    Calculates accuracy for unsupervised image segmentation using Hungarian matching.
    
    Args:
        pred_labels (np.ndarray): Predicted segmentation labels (2D or 3D array).
        gt_labels (np.ndarray): Ground truth segmentation labels (2D or 3D array).
    
    Returns:
        float: Accuracy as a fraction of correctly labeled pixels.
    """
    # Reshape the arrays into 1D
    gt_labels_1d = gt_labels.reshape(-1)
    pred_labels_1d = pred_labels.reshape(-1)
    
    # Get unique ground truth and predicted labels
    unique_gt_labels = np.unique(gt_labels_1d)
    unique_pred_labels = np.unique(pred_labels_1d)
    
    # Initialize the cost matrix (rows: ground truth, columns: predicted)
    cost_matrix = np.zeros((len(unique_gt_labels), len(unique_pred_labels)))
    
    # Populate the cost matrix (negative intersection for Hungarian minimization)
    for i, gt_label in enumerate(unique_gt_labels):
        for j, pred_label in enumerate(unique_pred_labels):
            # Find the intersection (common pixels) between the labels
            gt_indices = np.where(gt_labels_1d == gt_label)[0]
            pred_indices = np.where(pred_labels_1d == pred_label)[0]
            intersection = len(np.intersect1d(gt_indices, pred_indices))
            
            # Use -intersection as cost (minimizing negative intersection maximizes accuracy)
            cost_matrix[i, j] = -intersection
    
    # Solve the assignment problem using the Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Create a mapping from predicted labels to ground truth labels
    pred_to_gt_mapping = {}
    for row, col in zip(row_indices, col_indices):
        pred_to_gt_mapping[unique_pred_labels[col]] = unique_gt_labels[row]
    
    # Relabel the predicted labels based on the mapping
    remapped_pred_labels = np.zeros_like(pred_labels_1d)
    for pred_label, gt_label in pred_to_gt_mapping.items():
        remapped_pred_labels[pred_labels_1d == pred_label] = gt_label
    
    # Calculate accuracy as the fraction of correctly labeled pixels
    accuracy = np.mean(remapped_pred_labels == gt_labels_1d)
    return accuracy


def calculate_nmi(pred_labels, gt_labels):
    """
    Calculates the Normalized Mutual Information (NMI) between two label sets.
    
    Args:
        pred_labels (np.ndarray): Predicted segmentation labels (2D or 3D array).
        gt_labels (np.ndarray): Ground truth segmentation labels (2D or 3D array).
        
    Returns:
        float: Normalized Mutual Information (NMI) score.
    """
    gt_labels_1d = gt_labels.reshape(-1)
    pred_labels_1d = pred_labels.reshape(-1)
    
    nmi = normalized_mutual_info_score(gt_labels_1d, pred_labels_1d)
    return nmi


def calculate_homogeneity_score(pred_labels, gt_labels):
    """
    Calculates the homogeneity score between two label sets.
    
    Args:
        pred_labels (np.ndarray): Predicted segmentation labels (2D or 3D array).
        gt_labels (np.ndarray): Ground truth segmentation labels (2D or 3D array).
        
    Returns:
        float: Homogeneity score.
    """
    gt_labels_1d = gt_labels.reshape(-1)
    pred_labels_1d = pred_labels.reshape(-1)
    
    homogeneity = homogeneity_score(gt_labels_1d, pred_labels_1d)
    return homogeneity


def show_segementation(image_path, labels, palette = None, gt_path = None):
    """
    Display an image with segmentation labels overlaid using a color palette.
    
    Args:
        image_path (str): Path to the input image file.
        labels (np.ndarray): Segmentation labels (2D or 3D array).
        palette (dict): Color palette mapping label values to RGB colors.
        gt_path (str): Path to the ground truth segmentation image file (optional).
    """
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Generate a random color palette if not provided
    if palette is None:
        unique_labels = np.unique(labels)
        palette = {tuple(np.random.randint(0, 256, 3)): label for label in unique_labels}
    
    # Create a blank image for the segmentation overlay
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Map the labels to RGB colors using the palette
    for rgb, label in palette.items():
        mask = labels == label
        overlay[mask] = rgb
    
    # Blend the overlay with the original image
    alpha = 0.5
    output = cv2.addWeighted(image, alpha, overlay, 1 - alpha, 0)
    
    # Display the original image, segmentation overlay, and blended output
    cv2.imshow('Original Image', image)
    cv2.imshow('Segmentation Overlay', overlay)
    cv2.imshow('Segmentation Output', output)
    # save the output
    cv2.imwrite('output.png', output)
    cv2.imwrite('overlay.png', overlay)
    cv2.imwrite('image.png', image)

    if gt_path:
        gt_image = cv2.imread(gt_path)
        cv2.imshow('Ground Truth', gt_image)
        cv2.imwrite('gt.png', gt_image)
        gt_labels = convert_gt_to_labels(gt_path)
        remapped_labels = remap_labels(labels, gt_labels)
        for rgb, label in palette.items():
            mask = remapped_labels == label
            overlay[mask] = rgb
        print(f"Number of unique labels in ground truth: {len(np.unique(gt_labels))}")
        print(f"Number of unique labels in prediction: {len(np.unique(labels))}")
        print(f"Number of unique labels in remapped prediction: {len(np.unique(remapped_labels))}")
        cv2.imshow('Remapped Segmentation Overlay', overlay)
        cv2.imwrite('remapped_overlay.png', overlay)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def evaluate(pred_labels, gt_path):
    gt_labels = convert_gt_to_labels(gt_path) 
    miou = calculate_miou(pred_labels, gt_labels)
    accuracy = calculate_accuracy(pred_labels, gt_labels)
    nmi = calculate_nmi(pred_labels, gt_labels)
    homogeneity_score = calculate_homogeneity_score(pred_labels, gt_labels)
    return miou, accuracy, nmi, homogeneity_score
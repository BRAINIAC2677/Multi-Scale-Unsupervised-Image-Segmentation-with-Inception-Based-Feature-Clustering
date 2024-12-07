#from __future__ import print_function
import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn.init
import torch.nn.functional as F
from train import predict
from evaluate import evaluate, show_segementation

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--scribble', action='store_true', default=False, 
                    help='use scribbles')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--nEvalImg', metavar='NE', default=20, type=int, 
                    help='number of images to be evaluated')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, 
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                    help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                    help='visualization flag')
parser.add_argument('--input', metavar='DIRECTORY',
                    help='input image directory name', required=True)
parser.add_argument('--gt', metavar='DIRECTORY',
                    help='ground truth image directory name', required=True)
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                    help='step size for similarity loss', required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float, 
                    help='step size for continuity loss')
parser.add_argument('--stepsize_scr', metavar='SCR', default=0.5, type=float, 
                    help='step size for scribble loss')
args = parser.parse_args()



if __name__ == '__main__':
    config = vars(args).copy()
    mious = []
    accuracies = []

    if args.scribble:
        for img_filename in os.listdir(args.input):
            if img_filename.endswith('.jpg'):
                gt_filename = img_filename.split('.')[0] + '_gt.png'
                config['input'] = os.path.join(args.input, img_filename)
                config['gt'] = os.path.join(args.gt, gt_filename)
                pred_labels = predict(config)
                miou = evaluate(pred_labels, config['gt'])
                print(f'{img_filename}: {miou}')
                mious.append(miou)
    else:
        img_filenames = os.listdir(args.input)
        # img_filenames = np.random.choice(img_filenames, args.nEvalImg, replace=False)
        img_filenames = img_filenames[0:args.nEvalImg]
        for i in range(len(img_filenames)):
            img_filename = img_filenames[i]
            gt_filename = img_filename.split('.')[0] + '.png'
            config['input'] = os.path.join(args.input, img_filename)
            config['gt'] = os.path.join(args.gt, gt_filename)

            print(f'Evaluating {img_filename}...')
            pred_labels = predict(config)
            miou, accuracy = evaluate(pred_labels, config['gt'])
            mious.append(miou)
            accuracies.append(accuracy)
            print(f'\n{i} / {args.nEvalImg} | {img_filename} | miou: {miou} | accuracy: {accuracy}\n')
            print(f'Mean IoU: {np.mean(mious)}\n')
            print(f'Mean Accuracy: {np.mean(accuracies)}\n')
            show_segementation(config['input'], pred_labels, gt_path = config['gt'])

    for i in range(len(img_filenames)):
        print(f'{img_filenames[i]} | mIoU: {mious[i]} | Accuracy: {accuracies[i]}')
    print(f'\nMean IoU: {np.mean(mious)}\n')
    print (f'Mean Accuracy: {np.mean(accuracies)}\n')
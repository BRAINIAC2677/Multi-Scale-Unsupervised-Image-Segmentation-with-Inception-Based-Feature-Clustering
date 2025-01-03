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

parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                    help='number of channels')
parser.add_argument('--model', metavar='MODEL', default='inception', type=str, 
                    help='name of the model')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--nEvalImg', metavar='NE', default=-1, type=int, 
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
parser.add_argument('--log', metavar='LOG', default='log.txt', type=str, 
                    help='log file path')

args = parser.parse_args()


if __name__ == '__main__':
    config = vars(args).copy()
    logfile = open(args.log, 'w')
    mious = []
    accuracies = []
    nmis = []
    homogeneities = []

    img_filenames = os.listdir(args.input)
    if args.nEvalImg == -1:
        args.nEvalImg = len(img_filenames)
    args.nEvalImg = min(args.nEvalImg, len(img_filenames))
    # img_filenames = np.random.choice(img_filenames, args.nEvalImg, replace=False)
    img_filenames = img_filenames[0:args.nEvalImg]
    # img_filenames = ["2008_007836.jpg"]
    for i in range(len(img_filenames)):
        img_filename = img_filenames[i]
        gt_filename = img_filename.split('.')[0] + '.png'
        config['input'] = os.path.join(args.input, img_filename)
        config['gt'] = os.path.join(args.gt, gt_filename)

        print(f'Evaluating {img_filename}...')
        pred_labels = predict(config)
        miou, accuracy, nmi, homogeneity_score = evaluate(pred_labels, config['gt'])
        mious.append(miou)
        accuracies.append(accuracy)
        nmis.append(nmi)
        homogeneities.append(homogeneity_score)
        print(f'\n{i+1} / {args.nEvalImg} | {img_filename} | miou: {miou} | accuracy: {accuracy} | nmi: {nmi} | homogeneity: {homogeneity_score}\n')
        print(f'Mean IoU: {np.mean(mious)} | Mean Accuracy: {np.mean(accuracies)} | Mean NMI: {np.mean(nmis)} | Mean Homogeneity: {np.mean(homogeneities)}')
        if config['visualize']:
            show_segementation(config['input'], pred_labels, gt_path = config['gt'])
        print('---------------------------------------------------------------------\n')

    for i in range(len(img_filenames)):
        print(f'{img_filenames[i]} | mIoU: {mious[i]} | Accuracy: {accuracies[i]} | NMI: {nmis[i]} | Homogeneity: {homogeneities[i]}')
        logfile.write(f'{img_filenames[i]} | mIoU: {mious[i]} | Accuracy: {accuracies[i]} | NMI: {nmis[i]} | Homogeneity: {homogeneities[i]}\n')
    print(f'\nMean IoU: {np.mean(mious)}')
    print (f'Mean Accuracy: {np.mean(accuracies)}')
    print (f'Mean NMI: {np.mean(nmis)}')
    print (f'Mean Homogeneity: {np.mean(homogeneities)}\n')
    logfile.write(f'\nMean IoU: {np.mean(mious)} | Mean Accuracy: {np.mean(accuracies)} | Mean NMI: {np.mean(nmis)} | Mean Homogeneity: {np.mean(homogeneities)}\n')
    logfile.close()
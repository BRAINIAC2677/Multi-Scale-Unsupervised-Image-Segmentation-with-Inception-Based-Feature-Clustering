#from __future__ import print_function

import cv2
import torch
import numpy as np
import torch.nn.init
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from evaluate import evaluate
from vanillacnn import VanillaCNN
from inception import InceptionNet


use_cuda = torch.cuda.is_available()

def gpu_memory_info():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # in GB
        cached = torch.cuda.memory_reserved(device) / 1024**3  # in GB
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # in GB

        print(f"\nAllocated GPU Memory: {allocated:.2f} GB")
        print(f"Cached GPU Memory: {cached:.2f} GB")
        print(f"Total GPU Memory: {total_memory:.2f} GB")
    else:
        print("CUDA is not available.")

def clear_gpu_cache():
    torch.cuda.empty_cache()
    print("GPU cache cleared.")


def load_image(image_path):
    im = cv2.imread(image_path)
    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
    if use_cuda:
        data = data.cuda()
    data = Variable(data)
    return data, im


def train(config):
    # gpu_memory_info()
    clear_gpu_cache()
    # gpu_memory_info()
    data, im = load_image(config['input'])

    # train
    if config['model'] == 'vanillacnn':
        print ("Using VanillaCNN")
        model = VanillaCNN( input_dim = data.size(1), nChannel = config['nChannel'], nConv = config['nConv'] )
    elif config['model'] == 'inception':
        print ("Using InceptionNet")
        model = InceptionNet( input_dim = data.size(1), nChannel = config['nChannel'], nConv = config['nConv'] )
    else: 
        raise ValueError('Model not found')

    if use_cuda:
        model.cuda()
    model.train()

    # similarity loss definition
    loss_fn = torch.nn.CrossEntropyLoss()

    # continuity loss definition
    loss_hpy = torch.nn.L1Loss(reduction = 'mean')
    loss_hpz = torch.nn.L1Loss(reduction = 'mean')

    HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], config['nChannel'])
    HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, config['nChannel'])
    if use_cuda:
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()
        
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)

    for batch_idx in range(config['maxIter']):
        # forwarding
        optimizer.zero_grad()
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, config['nChannel'] )

        outputHP = output.reshape( (im.shape[0], im.shape[1], config['nChannel']) )
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = loss_hpy(HPy,HPy_target)
        lhpz = loss_hpz(HPz,HPz_target)

        _ , target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))

        # loss 
        loss = config['stepsize_sim'] * loss_fn(output, target) + config['stepsize_con'] * (lhpy + lhpz)
            
        loss.backward()
        optimizer.step()

        pred_labels = im_target.reshape(im.shape[0], im.shape[1])
        miou, accuracy, nmi, homogeneity_score = evaluate(pred_labels, config['gt'])
        print(f'{batch_idx + 1} / {config["maxIter"]} | label num : {nLabels} | loss : {loss.item():.5f} | miou : {miou:.5f} | accuracy : {accuracy:.5f} | nmi : {nmi:.5f} | homogeneity : {homogeneity_score:.5f}')

        if nLabels <= config['minLabels']:
            break
    
    return model 


def predict(config):
    model = train(config)
    data, im = load_image(config['input'])
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, config['nChannel'] )
    _ , target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    pred_labels = im_target.reshape(im.shape[0], im.shape[1])
    return pred_labels
 



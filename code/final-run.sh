#!/bin/bash

# Define the common arguments
MAX_ITER=150
N_EVAL_IMG=-1

# Run 1: Inception model on Pascal VOC 2012 dataset
echo "Running Inception model on Pascal VOC 2012..."
python validation.py --input ../data/pascal-voc-2012/val_images \
                     --gt ../data/pascal-voc-2012/val_labels \
                     --model inception \
                     --maxIter $MAX_ITER \
                     --nEvalImg $N_EVAL_IMG \
                     --log ../logs/inception-pascal-log.txt

# Run 2: VanillaCNN model on Pascal VOC 2012 dataset
echo "Running VanillaCNN model on Pascal VOC 2012..."
python validation.py --input ../data/pascal-voc-2012/val_images \
                     --gt ../data/pascal-voc-2012/val_labels \
                     --model vanillacnn \
                     --maxIter $MAX_ITER \
                     --nEvalImg $N_EVAL_IMG \
                     --log ../logs/vanillacnn-pascal-log.txt

# Run 3: Inception model on COCO Stuff 2017 dataset
echo "Running Inception model on COCO Stuff 2017..."
python validation.py --input ../data/coco-stuff-2017/val_images \
                     --gt ../data/coco-stuff-2017/val_labels \
                     --model inception \
                     --maxIter $MAX_ITER \
                     --nEvalImg $N_EVAL_IMG \
                     --log ../logs/inception-coco-log.txt

# Run 4: VanillaCNN model on COCO Stuff 2017 dataset
echo "Running VanillaCNN model on COCO Stuff 2017..."
python validation.py --input ../data/coco-stuff-2017/val_images \
                     --gt ../data/coco-stuff-2017/val_labels \
                     --model vanillacnn \
                     --maxIter $MAX_ITER \
                     --nEvalImg $N_EVAL_IMG \
                     --log ../logs/vanillacnn-coco-log.txt

echo "All runs completed!"

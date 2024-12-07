# Unsupervised Segmentation

### Dataset
- [PASCAL VOC 2012 Segmentation](https://www.kaggle.com/datasets/sovitrath/voc-2012-segmentation-data)

### Running Validation
```
python validation.py --input ../data/pascal-voc-2012-segmentation/train_images --gt ../data/pascal-voc-2012-segmentation/train_labels --maxIter 5 --nEvalImg 5
```

### Relevant Papers
- [Unsupervised Image Segmentation by Backpropagation](https://ieeexplore.ieee.org/abstract/document/8462533)
- [Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering](https://arxiv.org/pdf/2007.09990)
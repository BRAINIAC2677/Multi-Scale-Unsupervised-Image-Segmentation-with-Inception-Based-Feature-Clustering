# Multi-Scale Unsupervised Image Segmentation with Inception-Based Feature Clustering

### Abstract
his study presents an unsupervised image segmentation framework leveraging an InceptionNet-inspired architecture to cluster image pixels without labeled data. By integrating multi-scale feature extraction with differentiable clustering, the proposed method addresses the challenges of feature representation, spatial coherence, and adaptability in diverse datasets. Evaluated on PASCAL VOC 2012 and COCO-Stuff datasets, the model demonstrates robust segmentation performance, overcoming limitations of traditional methods reliant on hand-crafted features or fixed boundaries. The framework’s end-to-end optimization and clustering approach mark a significant step toward scalable and efficient unsupervised segmentation.

### Running Demo

- Install the following packages in your system.
    ```
    opencv-python
    numpy
    torch
    scipy
    scikit-learn
    Pillow
    argparse
    ```
- Clone the repository and move to `code` directory.
- Run the following command. You can change the passing arguments as you wish.
    ```
    python validation.py --input ../demo-data/pascal-voc-2012/val_images --gt ../demo-data/pascal-voc-2012/val_labels --maxIter 150 --nEvalImg 5 --visualize 0
    ```

### Contact Information
- [Asif Azad](asifazad0178@gmail.com)
- [Wasif Hamid](wsf.hmd99@gmail.com)
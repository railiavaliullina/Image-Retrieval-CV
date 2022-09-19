# Image-Retrieval

## About The Project

1) Implementation of training pipeline for Image Retrieval task,

2) Training on Stanfords Online Products Dataset (SOP).

## Getting Started

File to run:

        train/main.py
        
        or 
        
        hw-imageretrieval-final.ipynb

## Implementation details

Model architecture consists of:

    - ResNet-50 (without last classification layer) from paper 'Deep Residual Learning for Image Recognition'
    
            https://arxiv.org/pdf/1512.03385.pdf,
        
    - Global Average Pooling (GAP) applied to the last layer of ResNet,
    - fully-connected layer to get embedding for input image.

Loss Function and hard-negative sampling are from `FaceNet: A Unified Embedding for Face Recognition and Clustering` paper:
        
            https://arxiv.org/pdf/1503.03832.pdf
    

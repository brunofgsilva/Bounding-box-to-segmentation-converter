YOLOv8 Bounding Box to Segmentation Converter
This repository provides a Python script that converts YOLOv8 bounding box annotations into YOLOv8-compatible segmentation labels. The tool is designed to enhance segmentation workflows by leveraging bounding box annotations, enabling better model training for tasks requiring pixel-level precision.
Goal
The goal of this repository is to facilitate the conversion of existing YOLOv8 bounding box labels into segmentation labels.
Folder Structure
To use this script effectively, organize your project directory as follows:

project/
├── images/                 # Directory containing input images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels/                 # Directory containing YOLOv8 bounding box annotations
│   ├── image1.txt          
│   ├── image2.txt          
│   └── ...
├── SAM_models/                 # Directory containing SAM models
│   ├── sam_vit_l.pth     
└── convertion.py           # The Python script

Requirements
- Install SAM ( Follow the installation steps at https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints );
- In the same link, choose which model to install. I installed vit_l. In the script, it is a customisable variable;

# DHGCN: Dynamic Hop Graph Convolution Network for Self-supervised Point Cloud Learning

This is the source code for the implementation of our **DHGCN** for self-supervised point cloud learning.

<img src="./figure/overview.jpg" width="800" />

## Requirements

* CUDA = 11.3
* Python = 3.10
* PyTorch = 1.12.1
* Numpy = 1.23.1
* Package: h5py, sklearn, plyfile, json, cv2, yaml, tqdm

Different backbone networks for different tasks may have different requirements.

## Contents

We provide source codes for different point cloud learning tasks:

* [3D Object Classification](./obj_cls) on ModelNet40 and ScanObjectNN:
  - Unsupervised evaluation with different backbones (DGCNN, AdaptConv) with linear classifier on ModelNet40. 
  - Unsupervised evaluation with DGCNN as backbone with linear classifier on real-world dataset ScanObjectNN. 
  - Multi-task learning with different backbones (DGCNN, AdaptConv and PRA-Net) on ModelNet40.
  - Multi-task learning with DGCNN as the backbone on ScanObjectNN.
* [Shape Part Segmentation](./part_seg) on ShapeNet Part:
  - Unsupervised evaluation with PAConv as the backbone. 
  - Multi-task learning with different backbones (DGCNN and PAConv).

* [Scene Segmentation](./scene_seg) on S3DIS.

Please refer to the directory of each task for the usage details of our code.

## Results

#####  3D Object Classification
* Unsupervised Representation Learning

  - ModelNet40

  | Methods | Pretrained dataset |  Accuracy |
  | :---: |:---: | :---: |
  | DHGCN (DGCNN) | ShapeNet | 93.2 |
  | DHGCN (AdaptConv) | ShapeNet | 93.2 |
  | DHGCN (DGCNN) | ModelNet40 | 93.0 |
  | DHGCN (AdaptConv) | ModelNet40 | 93.3|

  - ScanObjectNN

  | Methods | OBJ_ONLY | OBJ_BG | PB_T50_RS|
  | :---: |:---: |:---: |:---: |
  | DHGCN (DGCNN) | 85.0 | 85.9 | 81.9 |


* Multi-task Learning

  - ModelNet40

  | Methods |  Accuracy |
  | :---: |:---: |
  | PRA-Net | 93.2 |
  | DHGCN (PRA-Net) | 93.4 (+0.2) |
  | DGCNN | 92.9 |
  | DHGCN (DGCNN) | 93.5 (+0.6) |
  | AdaptConv | 93.5 |
  | DHGCN (AdaptConv) | 93.6 (+0.2) |

  - ScanObjectNN

  | Methods | OBJ_ONLY | OBJ_BG | PB_T50_RS|
  | :---: |:---: |:---: |:---: |
  | DGCNN | 86.2 | 82.8 | 78.2|
  | DHGCN (DGCNN) | 88.3 (+2.1) | 88.3 (+5.5) | 82.9 (+4.8) |


#####  Shape Part Segmentation
* Unsupervised Representation Learning

| Methods | Class mIoU | Instance mIoU |
| :---: | :---: | :---: |
| DHGCN (PAConv) | 82.9 | 84.9 |

* Multi-task Learning

| Methods | Class mIoU | Instance mIoU |
| :---: | :---: | :---: |
| DGCNN | 82.3 | 85.2 |
| DHGCN (DGCNN) | 82.8 (+0.5) | 85.5 (+0.3) |
| PAConv | 84.2 | 86.0 |
| DHGCN (PAConv) | 84.5 (+0.3) |86.1 (+0.1)|


#####  Scene Segmentation
| Methods | Accuracy | Instance mIoU |
| :---: | :---: | :---: |
| DGCNN | 84.1 | 56.1 |
| DHGCN (DGCNN) | 86.4 (+2.3) | 61.7 (+5.6) |


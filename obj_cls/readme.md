# DHGCN for 3D Object Classification


## Unsupervised evaluation on ModelNet40

### Dataset

* For the self-supervised pretraining, we use the official dataset of [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and [ShapeNet](https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip). For ShapeNet dataset, following previous works, we use the prepared data in HDF5 files of ShapeNet and only use the coordinates of 2048 point as input. You need to download and place it to `./data/shapenet_part_seg_hdf5_data`.


* Download our pre-proceesed voxelized indices of points and hop distance matrices from [here](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/EpA1MpHiT-ZDoYoeCJVjDwQBZL144TS-MomertUdfx5JRg?e=HVNJZG), and place them to `./data`. You can also generate them automatically without the pre-processed files.

### Usage

* Firstly, you need to pretrain the model only with our hop distance reconstruction task on the pretrained dataset (i.e., ModelNet40 or ShapeNet):
  
    - Pretrain on ModelNet40:

    ~~~
    python train_pretrain.py --exp_name DHGCN_pretrain_AdaptConv_modelnet --dataset modelnet40 --num_points 1024 --backbone AdaptConv
    ~~~

    - Pretrain on ShapeNet:

    ~~~
    python train_pretrain.py --exp_name DHGCN_pretrain_AdaptConv_shapenet --dataset shapenet --num_points 2048 --backbone AdaptConv
    ~~~

    You can replace the backbone network by `--backbone`.

* After the pretraining stage, you need to load the pretrained model and freeze them to train a linear classifier on ModelNet40 by:

    ~~~
    python train_linear.py --exp_name DHGCN_linear_AdaptConv_modelnet --model_path 'path to load pretrained model on ModelNet40' --backbone AdaptConv
    ~~~
    ~~~
    python train_linear.py --exp_name DHGCN_linear_AdaptConv_shapenet --model_path 'path to load pretrained model on ShapeNet' --backbone AdaptConv
    ~~~

    You can also test our trained model (download [here](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/Ei4jJYlJvj5PqhLLEGPqLIIBg_zm4oEVzrma2K4EfRwvzA?e=P9HEfZ)) and put them to `./models`, then:

    ~~~
    python train_linear.py --exp_name DHGCN_linear_AdaptConv_modelnet --model_path 'models/pretrain_model_adapt_mn.t7' --backbone AdaptConv
    ~~~
    ~~~
    python train_linear.py --exp_name DHGCN_linear_AdaptConv_shapenet --model_path 'models/pretrain_model_adapt_sn.t7' --backbone AdaptConv
    ~~~

* Finally, you can test the linear classifier model by:

    ~~~
    python test_linear.py --model_path 'path to load trained linear classifier model' --backbone AdaptConv
    ~~~

    You can also use our trained model (download [here](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/Ei4jJYlJvj5PqhLLEGPqLIIBg_zm4oEVzrma2K4EfRwvzA?e=P9HEfZ)) and put them to `./models`, then:

    ~~~
    python test_linear.py --model_path './models/model_linear_adapt_sn.t7' --backbone AdaptConv
    ~~~
    
    Note that you can replace the backbone network by `--backbone`.

## Unsupervised evaluation on ScanObjectNN

### Dataset

* We use [ShapeNet](https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip) as the pretraining dataset. Following previous works, we use the prepared data in HDF5 files of ShapeNet and only use the coordinates of 2048 point as input. You need to download and place it to `./data/shapenet_part_seg_hdf5_data`.


* Download our pre-proceesed voxelized indices of points and hop distance matrices from [here](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/EpA1MpHiT-ZDoYoeCJVjDwQBZL144TS-MomertUdfx5JRg?e=HVNJZG), and place them to `./data`. You can also generate them automatically without the pre-processed files.

### Usage

* Firstly, you need to pretrain the model only with our hop distance reconstruction task on the pretrained dataset ShapeNet:
  
    ~~~
    python train_pretrain.py --exp_name DHGCN_pretrain_DGCNN_shapenet --dataset shapenet --num_points 2048
    ~~~
    
* After the pretraining stage, you need to load the pretrained model and freeze them to train a linear classifier on ScanObjectNN by:

  The ScanObjectNN dataset has three variants: OBJ_ONLY, OBJ_BG and PB_T50_RS. You can choose different variants by `--dataset`.

    - To train on OBJ_ONLY variant:

    ~~~
  python train_linear_scanobj.py --exp_name DHGCN_train_linear_objonly --dataset ScanObjectNN_objectonly
    ~~~

    - To train on OBJ_BG variant:
    ~~~
  python train_linear_scanobj.py --exp_name DHGCN_train_linear_objbg --dataset ScanObjectNN_objectbg
    ~~~

    - To train on PB_T50_RS variant:
    ~~~
  python train_linear_scanobj.py --exp_name DHGCN_train_linear_hardest --dataset ScanObjectNN_hardest --model_path 'path to load pretrained model on ShapeNet'
    ~~~
  You can also use our trained model (download [here](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/Ei4jJYlJvj5PqhLLEGPqLIIBg_zm4oEVzrma2K4EfRwvzA?e=P9HEfZ)) and put them to `./models`, then:
  
    ~~~
  python train_linear_scanobj.py --exp_name DHGCN_train_linear_objbg -- dataset ScanObjectNN_objectbg --model_path 'models/pretrain_model_dgcnn_sn.t7'
    ~~~
  
  Note that, you can choose different variants by `--dataset`.
  
* Finally, you can test the linear classifier model by:

    ~~~
    python test_linear_scanobj.py -- dataset ScanObjectNN_objectbg --model_path 'path to load trained linear classifier model'
    ~~~

    You can also test our trained model (download [here](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/Ei4jJYlJvj5PqhLLEGPqLIIBg_zm4oEVzrma2K4EfRwvzA?e=P9HEfZ) and put them to `./models`, then:

    ~~~
    python test_linear_scanobj.py -- dataset ScanObjectNN_objectbg --model_path './models/model_linear_scanobj_bg.t7'
    ~~~
    
    You can also choose different variants.

## Multi-task learning on ModelNet40

### Dataset

* Download and unzip the [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) from the official link, and place it to `./data/modelnet40_ply_hdf5_2048`. Following previous works, we use the prepared data in HDF5 files, where each object is already sampled to 2048 points. The experiments presented in the paper uses 1024 points for training and testing.

* Download our pre-proceesed voxelized indices of points and hop distance matrices from [here](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/EpA1MpHiT-ZDoYoeCJVjDwQBZL144TS-MomertUdfx5JRg?e=HVNJZG), and place them to `./data`. You can also generate them automatically without the pre-processed files.

### Usage

* Train a DHGCN model with multi-task learning (hop distance reconstruction task and classification task) on the ModelNet40 dataset:

    - With DGCNN as the backbone:

    ~~~
    python train.py --exp_name DHGCN_train_DGCNN --backbone DGCNN
    ~~~

    - With AdaptConv as the backbone:
    ~~~
    python train.py --exp_name DHGCN_train_AdaptConv --backbone AdaptConv
    ~~~

    - With PRA-Net as the backbone:
    ~~~
    python train.py --exp_name DHGCN_train_PRANet --backbone PRANet
    ~~~

    By default a single GPU is used: `0`, or you can specify  the GPU index by `--gpu`. The split number of voxelization is set to `3` as default  and you can change it by `--split_num`.

* After the training stage, you can test the model by:

    ~~~
    python test.py --model_path 'path to trained model' --backbone 'DGCNN/AdaptConv/PRANet'
    ~~~

    Or you can use our trained model (download [here](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/Ei4jJYlJvj5PqhLLEGPqLIIBg_zm4oEVzrma2K4EfRwvzA?e=P9HEfZ)) and put them to `./models`, then test the model by:

    ~~~
    python test.py --model_path './models/model_dgcnn.t7' --backbone DGCNN
    ~~~
    ~~~
    python test.py --model_path './models/model_adapt.t7' --backbone AdaptConv
    ~~~
    ~~~
    python test.py --model_path './models/model_pranet.t7' --backbone PRANet
    ~~~


## Multi-task learning on ScanObjectNN

### Dataset

* Download and unzip the official dataset of [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/), and place it to `./data/ScanObjectNN`.

* Download our pre-proceesed voxelized indices of points and hop distance matrices from [here](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/EpA1MpHiT-ZDoYoeCJVjDwQBZL144TS-MomertUdfx5JRg?e=HVNJZG) and place them to `./data`. You can also generate them automatically without the pre-processed files.

### Usage

* Train a DHGCN (DGCNN as the backbone) model with multi-task learning on the ScanObjectNN dataset:

    The ScanObjectNN dataset has three variants: OBJ_ONLY, OBJ_BG and PB_T50_RS. You can choose different variants by `--dataset`.

    - To train on OBJ_ONLY variant:

    ~~~
    python train_scanobj.py --exp_name DHGCN_train_objonly --dataset ScanObjectNN_objectonly
    ~~~

    - To train on OBJ_BG variant:
    ~~~
    python train_scanobj.py --exp_name DHGCN_train_objbg --dataset ScanObjectNN_objectbg
    ~~~

    - To train on PB_T50_RS variant:
    ~~~
    python train_scanobj.py --exp_name DHGCN_train_hardest --dataset ScanObjectNN_hardest
    ~~~

* After the training stage, you can test the model by:

    ~~~
    python test_scanobj.py --model_path 'path to trained model' --dataset 'variant you used'
    ~~~

    Or you can also use our trained model (download [here](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/Ei4jJYlJvj5PqhLLEGPqLIIBg_zm4oEVzrma2K4EfRwvzA?e=P9HEfZ)) and put them to `./models`, then test the model by:

    ~~~
    python test_scanobj.py --model_path './models/model_scanobj_only.t7' --dataset ScanObjectNN_objectonly
    ~~~
    ~~~
    python test_scanobj.py --model_path './models/model_scanobj_bg.t7' --dataset ScanObjectNN_objectbg
    ~~~
    ~~~
    python test_scanobj.py --model_path './models/model_scanobj_hardest.t7' --dataset ScanObjectNN_hardest
    ~~~

# DHGCN (DGCNN as the backbone) for Shape Part Segmentation

### Dataset
* Download and unzip the ShapeNet Part dataset (xyz, normals and labels) from the official [link](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip), and place it to `./data/shapenetcore_partanno_segmentation_benchmark_v0_normal`. 

* Download our pre-proceesed voxelized indices of points and hop distance matrices from [here](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/EidipqeItd1KkV3zk3WhsNMBZxxtZvLDrzjysGaPgC32JA?e=3igVSC), and place them to `./data`. You can also generate them automatically without the pre-processed files.

### Usage

* Train a DHGCN model with multi-task learning (hop distance reconstruction task and shape part segmentation task) on the ShapeNet Part dataset:

    ~~~
    python train.py --name DHGCN_train
    ~~~

    By default two GPUs are used: `[0, 1]`, or you can specify  the GPU index by `--gpu`. The split number of voxelization is set to `5` as default  and you can change it by `--split_num`.


* Test the model from `models/exp_name/checkpoints/`:

    ~~~
    python test.py  --name DHGCN_test --model_path exp_name --checkpoint model_best_c_miou.pkl
    ~~~

    Or you can also use our trained model (download [here](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/EsnpMbJoebRBmSWc-20oEcABcoNRAYwlNEWLFPlj54zmQg?e=XyK7SO)) and put them to `./models`, then test the model by:

    ~~~
    python test.py  --name DHGCN_test --model_path DHGCN_train --checkpoint model_best_c_miou.pkl
    ~~~

* For visualization, you can save the predicted obj files (ground truth, prediction, difference):

    ~~~
    python test.py --name DHGCN_test --model_path DHGCN_train --checkpoint model_best_c_miou.pkl --vis_dir ./vis_results
    ~~~
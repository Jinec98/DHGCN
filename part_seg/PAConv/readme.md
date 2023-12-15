# DHGCN (PAConv as the backbone) for Shape Part Segmentation

### Dataset
* Download and unzip the ShapeNet Part dataset (xyz, normals and labels) from the official [link](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip), and place it to `./data/shapenetcore_partanno_segmentation_benchmark_v0_normal`. 

* Download our pre-proceesed voxelized indices of points and hop distance matrices from [here](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/EnqjVZGXVQpIt70ZS36qxbkBfM-ddnpV1L8TDTIvEHMVvg?e=ZKDF1e), and place them to `./data`. You can also generate them automatically without the pre-processed files.

### Usage

* In order to use PAConv as the backbone, you need to wait a while to compile PAConv's [cuda_lib](./cuda_lib/) at the first run. 


* We conduct unsupervised representation learning on shape part segmentation. 

    First, you need to pretrain the model only with our self-supervised task:

    We use the PAConv multi-process training version to reduce the training time, so you need to specify GPU index:
    
    ```
    CUDA_VISIBLE_DEVICES=0,1 python train_pretrain.py --config config/unsup_pretrain.yaml
    ```
    
    Then, load the pretrained model and freeze them to train several MLPs:
    
    ```
    CUDA_VISIBLE_DEVICES=0,1 python train_linear.py --config config/unsup_linear.yaml
    ```
    
    You can test the trained model by:
    
    ```
    python test_linear.py --config config/unsup_test.yaml
    ```
    
    You can also test our trained model (download [here](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/Ej9dyo_lYDdDl666wjHijcsBbDvB084W0qHJ4nuQ9nL-lQ?e=g7R103)) and put them to `./models`. The trained model in pretraining stage named `DHGCN_pretrain`, and the trained model with linear evaluation named `DHGCN_linear`. You need to modify the `model_name` in corresponding config files in order to load correct models.


* Further, we train a DHGCN model with multi-task learning on the ShapeNet Part dataset:

    ```
    CUDA_VISIBLE_DEVICES=0,1 python train.py --config config/train.yaml
    ```
    
    Test the model without any voting strategy:

    ```
    python test.py --config config/test.yaml
    ```
    
    
    * You can set the `model_name` in the config file to specify the trained model in `./checkpoints`(e.g., `DHGCN_train`). And you can choose to test the model with different metrics, by specifying `model_type` to `insiou`, `clsiou` or `acc`.
    * You can also test our trained model (download [here](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/Ej9dyo_lYDdDl666wjHijcsBbDvB084W0qHJ4nuQ9nL-lQ?e=g7R103)) and put them to `./models`,  then set the `model_name` to `DHGCN_train` in the test config file.
    * For visualization, you can save the predicted results by setting the `vis_dir`, e.g., `./vis_results`. Then the point clouds' obj files will be saved automatically.

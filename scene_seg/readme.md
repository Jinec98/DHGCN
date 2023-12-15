# DHGCN (DGCNN as the backbone) for Scene Segmentation

### Dataset
* Download and unzip the S3DIS dataset (named `Stanford3dDataset_v1.2_Aligned_Version.zip`) from the official [link](https://goo.gl/forms/4SoGp4KtH1jfRqEj2), and place it to `./data/Stanford3dDataset_v1.2_Aligned_Version`. 

* Download our pre-proceesed voxelized indices of points and hop distance matrices from [here](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/Ejal05xw8cZAvMWUGcgNZAwB76heCEd9HXXOoUHSfRm1Fg?e=v919WK) and place them to `./data`. You can also generate them automatically without the pre-processed files.

### Usage

This task uses 6-fold cross validation, 1 of the 6 areas will be selected as the testing area and the remaining 5 areas will be trained, eventually producing 6 models.

* To train in area 1-5 and test in area 6:

    ~~~
    python train.py --exp_name DHGCN_train_6 --test_area 6
    ~~~

    Then, change the area to test:
    ~~~
    python train.py --exp_name DHGCN_train_i --test_area 1/2/3/4/5/6
    ~~~

    Finally, you can get 6 models.

* To test the model from `outputs/exp_name/models/`:

    For example, test in area 6 after the model is trained in area 1-5:

    ~~~
    python test.py --exp_name DHGCN_test_6 --test_area 6 --model_path outputs/DHGCN_train_6/models/
    ~~~

    And, you can test in all areas after 6 models are trained. The trained models need to be placed in the same path and renamed according to the test area index, e.g., `model_1.t7`, `model_2.t7`, etc.

    ~~~
    python test.py --exp_name DHGCN_test_all --test_area all --model_path outputs/DHGCN_all_models/models/
    ~~~

    You can also test our trained model (download [here](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/EnUatNEzCXlClQ2pcDCnWr8B1DlXZSF18oQFDcoLZqXCig?e=BaInHk)) and put them to `./models`.

    ~~~
    python test.py --exp_name DHGCN_test_all --test_area all --model_path models/
    ~~~


* For visualization, you can save the predicted results by ply files:

    ~~~
    python test.py --exp_name DHGCN_test_all --test_area all --model_path models/ --visu=all --visu_format=ply
    ~~~

    You can use `--vis` to specify the visualization file, which is the same as used in DGCNN.
# Ultralytics YOLO11-pose Multi-Label


## Abstract
### Versions
| Component | Version     |
|-----------|-------------|
| Python    | 3.8.10      |
| Torch     | 2.4.1+cu121 |
| CUDA      | 12.1        |
| cuDNN     | 9.1.0       |
|Ultralytics|8.3.111|

<hr/>

## Description

### Dataset Preparation
- Dataset label file must follow the following format:
    ```
  [LBL_0] [LBL_1] ... [LBL_N] [BBOXES] [KEYPOINTS]
    ```
  , for example:
    ```
  4 2 1 0 1 0 0.276333 0.541833 0.120667 0.074333 0.219333 0.527667 1.000000 0.224333 0.533667 1.000000 0.239333 0.534333 1.000000 0.239333 0.522667 1.000000 0.240333 0.543333 1.000000 0.256000 0.508000 1.000000 0.246333 0.535333 1.000000 0.255000 0.565333 1.000000 0.239333 0.550333 1.000000 0.273333 0.531333 1.000000 0.266667 0.552000 1.000000 0.286000 0.525333 1.000000 0.311333 0.549667 1.000000 0.295667 0.565667 1.000000 0.333333 0.575667 1.000000
  6 2 1 0 1 0 0.305333 0.369167 0.116000 0.073000 0.250667 0.369000 1.000000 0.263000 0.376000 0.000000 0.265667 0.368000 0.000000 0.261667 0.349333 1.000000 0.279000 0.383000 0.000000 0.275000 0.364000 1.000000 0.267667 0.392000 1.000000 0.284667 0.402333 0.000000 0.261667 0.385000 1.000000 0.298667 0.336000 0.000000 0.298000 0.363000 0.000000 0.319333 0.358667 0.000000 0.356000 0.337667 0.000000 0.322667 0.379667 0.000000 0.360000 0.361000 0.000000
  4 2 1 0 1 0 0.315167 0.695833 0.133000 0.055667 0.252000 0.687333 1.000000 0.260667 0.692000 1.000000 0.270667 0.690667 1.000000 0.271000 0.674333 1.000000 0.276000 0.697667 1.000000 0.287667 0.671333 1.000000 0.294333 0.691667 1.000000 0.288667 0.701333 1.000000 0.289667 0.693000 1.000000 0.313667 0.687000 0.000000 0.306000 0.706667 0.000000 0.340333 0.691667 0.000000 0.374667 0.704667 0.000000 0.339000 0.718000 0.000000 0.378333 0.720333 1.000000
  6 2 1 0 1 0 0.663167 0.706833 0.114333 0.069000 0.717000 0.696333 1.000000 0.709000 0.697333 1.000000 0.699667 0.702333 1.000000 0.708000 0.715333 1.000000 0.692667 0.696333 0.000000 0.694000 0.722667 1.000000 0.699667 0.696667 1.000000 0.686333 0.675667 0.000000 0.705000 0.692333 1.000000 0.674333 0.733000 0.000000 0.670333 0.709333 0.000000 0.640000 0.719667 0.000000 0.611333 0.738000 0.000000 0.640000 0.703667 0.000000 0.609333 0.729667 0.000000
  1 1 2 0 2 0 0.640833 0.420333 0.150333 0.118000 0.712667 0.376000 1.000000 0.689333 0.371333 0.000000 0.644667 0.391000 1.000000 0.650667 0.364667 1.000000 0.652000 0.424000 1.000000 0.609333 0.365667 1.000000 0.592667 0.370667 0.000000 0.663000 0.476000 1.000000 0.691667 0.434667 1.000000 0.598000 0.414000 1.000000 0.610000 0.446333 1.000000 0.585333 0.430000 1.000000 0.569000 0.444000 1.000000 0.593000 0.451667 1.000000 0.572667 0.458667 1.000000
  7 1 1 0 1 0 0.700500 0.498167 0.159667 0.136333 0.777000 0.550333 1.000000 0.743333 0.540333 1.000000 0.757667 0.518000 1.000000 0.733333 0.563000 0.000000 0.736000 0.491667 1.000000 0.716333 0.521000 1.000000 0.687333 0.545000 1.000000 0.705667 0.467000 1.000000 0.674333 0.493000 1.000000 0.741333 0.451667 0.000000 0.698000 0.484667 1.000000 0.726333 0.433333 0.000000 0.670667 0.441333 0.000000 0.655000 0.487667 1.000000 0.624000 0.483667 1.000000
    ```
- You need to write your own dataset configuration file 
  - ```names```: Name of classes of each label, unified.
  - ```nc```: Number of total classes.
  - ```nl```: Number of total labels.
  - ```main_idx```: Index of main-label.
  - ```sub_idxs```: Indeces of sub-labels.
  - ```start_idxs```: Starting indeces of each label.
  - ```ncs```: Number of classes for each label.
  - ```kpt_shape```: Shape of keypoint, usually Nx3.
  - ```train```, ```test```, ```val```: Directory of train, test, validation splits.

### Model Preparation
- You need to modify model configuration file: ```yolo11-pose.yaml``` according to your dataset.
  - ```nc```: Number of total classes.
  - ```kpt_shape```: Shape of keypoint, usually Nx3.
  - ```scale```: Model scale.

### Parameters Setting
- You need to modify ```parameters.py``` according to your directory setting.
  - ```['img_path']```: Directory of your dataset.
  - ```['hyp'].data```: Name of your dataset configuration file.

### Training
- You need to modify ```train.py``` according to your settings.
  - ```./data_mmpatient_full.yaml'``` -> Name of your dataset configuration file.
    ```
      data = yaml.safe_load(open(f'./data_mmpatient_full.yaml', 'rb'))
    ```
  - ```f'./yolo11-pose.yaml'``` -> Name of your model configuration file.
    ```
      model = MMPatientYoloModel(f'./yolo11-pose.yaml')
    ```
- After the modification, run ```train.py``` to begin training.
  ```
  python train.py
  ```

<hr/>

## Reference

- Ultralytics
  ```
  @software{Jocher_Ultralytics_YOLO_2023,
  author = {Jocher, Glenn and Qiu, Jing and Chaurasia, Ayush},
  license = {AGPL-3.0},
  month = jan,
  title = {{Ultralytics YOLO}},
  url = {https://github.com/ultralytics/ultralytics},
  version = {8.0.0},
  year = {2023}
  }
  ```
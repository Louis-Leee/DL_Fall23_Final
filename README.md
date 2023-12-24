# DL23-FINAL: Vision-centric 3D Semantic Occupancy Prediction for Autonomous Driving
Sihang LI, Yiming Li 


## Requirements

1. Create conda environment:

```
$ conda create -y -n DESM python=3.7
$ conda activate DESM
```
2. This code is compatible with python 3.7, pytorch 1.7.1 and CUDA 10.2. Please install [PyTorch](https://pytorch.org/): 

```
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```

3. Install the additional dependencies:

```
$ cd desm/
$ pip install -r requirements.txt
```

4. Install tbb:

```
$ conda install -c bioconda tbb=2020.2
```

5. Downgrade torchmetrics to 0.6.0
```
$ pip install torchmetrics==0.6.0
```

6. Finally, install DESM:

```
$ pip install -e ./
```


## Datasets


### SemanticKITTI

1. You need to download

      - The **Semantic Scene Completion dataset v1.1** (SemanticKITTI voxel data (700 MB)) from [SemanticKITTI website](http://www.semantic-kitti.org/dataset.html#download)
      -  The **KITTI Odometry Benchmark calibration data** (Download odometry data set (calibration files, 1 MB)) and the **RGB images** (Download odometry data set (color, 65 GB)) from [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).


2. Create a folder to store SemanticKITTI preprocess data at `/path/to/kitti/preprocess/folder`.

3. Store paths in environment variables for faster access (**Note: folder 'dataset' is in /path/to/semantic_kitti**):

```
$ export KITTI_PREPROCESS=/path/to/kitti/preprocess/folder
$ export KITTI_ROOT=/path/to/semantic_kitti 
```

4. Preprocess the data to generate labels at a lower scale, which are used to compute the ground truth relation matrices:

```
$ python desm/data/semantic_kitti/preprocess.py kitti_root=$KITTI_ROOT kitti_preprocess_root=$KITTI_PREPROCESS
```


## Pretrained models

Download DESM pretrained models on [SemanticKITTI](https://drive.google.com/file/d/1tc9aWV9G4hQ2q6_XUK3ZB4gXMW09P2mw/view?usp=sharing).

## Training

To train DESM with SemanticKITTI, type:

### SemanticKITTI

1. Create folders to store training logs at **/path/to/kitti/logdir**.

2. Store in an environment variable:

```
$ export KITTI_LOG=/path/to/kitti/logdir
```

3. Train DESM using 4 GPUs with batch_size of 4 (1 item per GPU) on Semantic KITTI:

```

$ python desm/scripts/train_desm.py \
    dataset=kitti \
    enable_log=true \
    kitti_root=$KITTI_ROOT \
    kitti_preprocess_root=$KITTI_PREPROCESS\
    kitti_logdir=$KITTI_LOG \
    n_gpus=4 batch_size=4    
```




## Evaluating 

### SemanticKITTI

To evaluate DESM on SemanticKITTI validation set, type:

```

$ python desm/scripts/eval_desm.py \
    dataset=kitti \
    kitti_root=$KITTI_ROOT \
    kitti_preprocess_root=$KITTI_PREPROCESS \
    n_gpus=1 batch_size=1
```



## Inference

Please create folder **/path/to/desm/output** to store the DESM outputs and store in environment variable:

```
export MONOSCENE_OUTPUT=/path/to/desm/output
```



### Semantic KITTI

To generate the predictions on the Semantic KITTI validation set, type:

```

$ python desm/scripts/generate_output.py \
    +output_path=$MONOSCENE_OUTPUT \
    dataset=kitti \
    kitti_root=$KITTI_ROOT \
    kitti_preprocess_root=$KITTI_PREPROCESS \
    n_gpus=1 batch_size=1
```


## Visualization




We use mayavi to visualize the predictions. Please install mayavi following the [official installation instruction](https://docs.enthought.com/mayavi/mayavi/installation.html). Then, use the following commands to visualize the outputs on respective datasets.


You also need to install some packages used by the visualization scripts using the commands:
```
pip install tqdm
pip install omegaconf
pip install hydra-core
```

### Semantic KITTI 

```

$ python desm/scripts/visualization/kitti_vis_pred.py +file=/path/to/output/file.pkl +dataset=kitt
```



## Acknowledgement
This project is built based on MonoScene. We thank the contributors of the prior project for building such excellent codebase and repo. Please refer to this repo (https://github.com/astra-vision/MonoScene) for more documentations and details.


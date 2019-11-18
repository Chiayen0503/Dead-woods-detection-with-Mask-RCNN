# Dead woods detection with Mask RCNN 
* This repository holds my final year dissertation project. The project is extended from EU CanHeMon which suggested using neural network may help detect standing deadtrees. Link: https://ec.europa.eu/jrc/en/publication/canopy-health-monitoring-canhemon-project.    

* The project will help you build your own synthetic data with annotations and train a Mask RCNN model in a tensorflow docker. 

* You can also hire an EC2 accelerated computing instance which saves time on training models. For more information to set up AWS EC2 p3.2xlarge instance, please refer to: https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/

## Create your own synthetic datasets and corresponding coco-format annotation file
This will save you a lot of effort from manually making annotations through VGG annotator. The annotation process is always tedious and frustrative. However, you can use this synthetic method to prevent it. In addition, there is no limit the number of images you create.

* The synthetic method developed by Immersive Limit, whose founder is an expert in AI and image processing.   

### Step 1: Download repositery to local
```
git clone https://github.com/akTwelve/cocosynth.git
```
* Later, We will only use two files and a folder: 
* (1) image_composition.py:
    Synthesize images by assigning foregrounds (target Objects) to backgrounds and export an annotation file (json). 
* (2) coco_json_utils.py:
    Format the annotation file to follow COCO annotation style.  
* (3) datasets:
    A folder where stores input foregrounds and backgrounds, and outputs synthetic images.
### Step 2: Collect foregrounds (target object) and background images
* (1) Download Gimp: https://www.gimp.org/downloads/
* (2) See tutorial "How to cut out an object in Gimp": https://www.youtube.com/watch?v=DLryAXsIZ04 
* (3) Create folders in "dataset"; the folders follow tree below:

```
.
├── input
│   ├── backgrounds
│   └── foregrounds
│       └── super_category
│           └── category
└── output

```
* (3) puts target object pictures in "category" and background pictures in "background" folder.
* (4) rename "super_category" and "category" with your labels' names, you can have multiple "category" folders. Ex, rename super_category as pet, one category as dog and the other as cat. Vice versa, multiple "super_category" folders are acceptable.

### Step 3: Synthesize foregrounds (target object) and background images
```
cd cocosynth-master/python
python3 image_composition.py --input_dir <path to input> --output_dir <path to output> --count <number of synthetic images> --width <width of a synthetic image> --height <height of a synthetic image>

```
for example, 
```
python3 image_composition.py --input_dir /cocosynth-master/datasets/input --output_dir /cocosynth-master/datasets/output --count 5 --width 850 --height 850
# Note: mask rcnn prefer square images so you may set width equals to height to save your time from preprocessing your images.
```

following up adding data information in dataset_info.json:
```
Would you like to create dataset info json? (y/n) y
Note: you can always modify the json manually if you need to update this.
Description: 
URL: 
Version: 
Contributor: 
Add an image license? (y/n) y
License name: 
License URL:
```

### Step 4: Format the annotation file <mask_definitions.json> to follow COCO annotation style.
```
cd <path to output>
python3 coco_json_utils.py -md mask_definitions.json -di dataset_info.json
```
you will get a coco style annotation file <coco_instances.json>. In addition, synthetic images will be stored in "images" folder and corresponding masks will be stored in "masks" folder. 
* Your "output" tree may look like this : 
```
.
├── coco_instances.json
├── coco_json_utils.py
├── dataset_info.json
├── images
│   ├── 000000000.jpg
│   └── ...
├── mask_definitions.json
└── masks
    ├── 000000000.png
    └── ...
```
### Step 5: Redo Step 1 to Step 4 to build a second, synthetic-image dataset as validation dataset. 
* Copy all files from both outputs and seperately stores them into two folders named "train" and "val". Delete both "masks" folders, <dataset_info.json>, <coco_json_utils.py> and <mask_definitions.json> because Mask RCNN Matterport doesn't read them when training models. 
* Your train and val datasets are prepared!

```
.
├── train
│   ├── coco_instances.json
│   └── images
│       ├── 000000000.jpg
│       └── ...
└── val
    ├── coco_instances.json
    └── images
        ├── 000000000.jpg
        └── ...
``` 

## Set an environment that can run Mask RCNN model
### Step 0: You may login to EC2 p3.2xlarge instance or simply execute following instructions in local.
### Step 1: Open a new terminal window and download Docker
```
sudo apt-get install docker.io
```
### Step 2: Download and execute a docker image in the first terminal
Using a docker image which pre-installed tensorflow library, can prevent us dealing with tedious installation process and focus on training models or refining dataset. Here, we use Tensorflow docker image for training neural network models.  

Download and execute CPU-only image:
```
docker pull tensorflow/tensorflow
docker run -it -p 8888:8888 --rm tensorflow/tensorflow:latest-py3 bash
```
Download and execute GPU-only image:
```
docker pull tensorflow/tensorflow:latest-gpu-py3
nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:latest-gpu-py3 bash
```

### Step 3: Run requirement.txt to install other dependancies and Mask RCNN model

```
git clone https://github.com/matterport/Mask_RCNN.git
cd Mask_RCNN-master
pip3 install -r requirements.txt

```

## Train a Mask RCNN model on your synthetic dataset
* (1) Downloads this repositery (dissertation-master) and delete example train and val folders  
```
cd ~
git clone https://github.com/Chiayen0503/dissertation.git
cd dissertation-master/datasets 
sudo rm -f -r train
sudo rm -f -r val
```
* (2) Open a second terminal and check docker images ID

```
docker ps
```

* (3) Use the second terminal to upload your local train and val folders to docker images
```
cd /to/your/train/and/val/folders
nvidia-docker cp train <replace with container ID>:/dissertation-master/datasets/train
nvidia-docker cp val <replace with container ID>:/dissertation-master/datasets/train
```
example: (change nvidia-docker to docker if you're running CPU-only docker image)
```
nvidia-docker cp train 2c89b6975e72:/dissertation-master/datasets/train
```

* (3) Train model in your docker environment (first terminal)
```
cd /path/to/dissertation-master
python3 custom.py train --dataset=/dissertation/datasets --weights=coco
```

* (4) You will get a score.csv and a .h5 file. The former allows you to investigate loss functions; the latter allows you to visualize mask predictions on either synthetic or raw picture. Please see <inspect_loss_history.ipynb> and <model_prediction_visualization.ipynb>.

* (5) Samples of mask visualization please refer to visualization-samples 

![alt text](https://github.com/Chiayen0503/dissertation/blob/master/visualization-samples/01.png)
![alt text](https://github.com/Chiayen0503/dissertation/blob/master/visualization-samples/05.png)

## Extra to learn: Manipulate hyperparameters to fine tune mask rcnn
please check hyperparameters-fine-tune folder: https://github.com/Chiayen0503/dissertation/tree/master/hyperparameters-fine-tune

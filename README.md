# Dead woods detection with Mask RCNN 
This repository holds my final year in Lancaster University and dissertation project is extended from EU CanHeMon project.   

The project will help you build your own synthetic data with annotations and train a Mask RCNN model in a tensorflow docker. You can also hire an EC2 accelerated computing instance which saves time on training models. For more information to set up AWS EC2 p3.2xlarge instance, please refer to:

## Create your own synthetic datasets and a coco format annotation file
### Step 1: Download repositery 
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
* (3) fills example pictures in "category" and "background" folder with yours 
* (4) rename "super_category" and "category" with your labels' names, you can have multiple "category" folders. Ex, rename super_category as pet, one category as dog and the other as cat. Vice versa, multiple "super_category" folders are acceptable.

### Step 3: Synthesize foregrounds (target object) and background images
```
cd cocosynth-master/python
python3 image_composition.py --input_dir <path to input> --output_dir <path to output> --count <number of synthetic images> --width <width of a synthetic image> --height <height of a synthetic image>

```
for example, 
```
python3 image_composition.py --input_dir /input --output_dir /output --count 5 --width 850 --height 850
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
* (1) Your "output" tree may look like this : 
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
## Step 5: Redo Step 1 to Step 4 to build a second, synthetic-image dataset as validation dataset. 
* Copy all files from both outputs and seperately stores them into two folders named "train" and "val"
* Your dataset is prepared!
```
.
├── train
│   ├── coco_instances.json
│   ├── coco_json_utils.py
│   ├── dataset_info.json
│   ├── images
│   │   ├── 000000000.jpg
│   │   └── ...
│   ├── mask_definitions.json
│   └── masks
│       ├── 000000000.png
│       └── ...
└── val
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
In fact, you can even delete "masks" folders as Mask RCNN Matterport only read annotation file (json) and images.   

## Set the environment that can run Mask RCNN model
### Step 1: Download Docker
```
sudo apt-get install docker.io
```
### Step 2: Download and execute a docker image in terminal
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

### Step 3: Run a shell script to install other dependancies and Mask RCNN model that are not included in Tensorflow docker

```
git https://github.com/Chiayen0503/dissertation.git
cd dissertation
bash install.sh
```

## Train a Mask RCNN model on your synthetic dataset







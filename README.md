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
* (3) Create folders that follows the following tree:
*  dataset - input - foregrounds - super_category - category
                   - backgrounds
           - output

* (3) fills example pictures in "category" and "background" folder with yours 
* (4) rename "super_category" and "category" with your labels, you can have multiple categories. Ex, rename super_category as pet, category1 as dog and category2 as cat

### Step 3: Synthesize foregrounds (target object) and background images
```
cd cocosynth-master/python
python3 image_composition.py --input_dir <path to input> --output_dir <path to output> --count <number of synthetic images> --width <width of a synthetic image> --height <height of a synthetic image>

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
you will get a coco style annotation file <coco_instances.json>

## Set the environment that can run Mask RCNN model
### Step 1: Download Docker
```
sudo apt-get install docker.io
```
### Step 2: Download and execute a docker image in terminal
Using a docker file which pre-installed tensorflow library, can prevent us dealing with tedious installation process and focus on training models or refining dataset.  

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
* (1) open a new terminal window
```
docker cp src/. mycontainer:/target
```
* (2) 
```
git https://github.com/Chiayen0503/dissertation.git
cd dissertation
bash install.sh
```

## Train a Mask RCNN model on your synthetic dataset







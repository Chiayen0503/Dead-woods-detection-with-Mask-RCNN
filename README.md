# Dead woods detection with Mask RCNN 
This repository holds my final year in Lancaster University and dissertation project is extended from EU CanHeMon project.   

## Getting Started
The project will help you build your own synthetic data with annotations and train a Mask RCNN model in a tensorflow-gpu docker. You can also hire an EC2 accelerated computing instance which saves time on training models. For more information to set up AWS EC2 p3.2xlarge instance, please refer to:



### Download and execute a docker image from tensorflow 
Using a docker file which pre-installed tensorflow library, can prevent us dealing with tedious installation process and focus on training models or refining dataset.  

Download and execute CPU-only image:
```
docker pull tensorflow/tensorflow)
docker run -it -p 8888:8888 --rm tensorflow/tensorflow:latest-py3 bash 
```
Download and execute GPU-only image:
```
docker pull tensorflow/tensorflow:latest-gpu-py3
nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:latest-gpu-py3
```

### Write a shell script to install other dependancies that are not included in Tensorflow docker
Create a shell script:
```
nano install.sh
```

Run the shell script:
```
bash install.sh
```


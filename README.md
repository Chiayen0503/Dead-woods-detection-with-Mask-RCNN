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
nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:latest-gpu-py3 bash
```

### Write a shell script to install other dependancies that are not included in Tensorflow docker
Create a shell script:
```
nano <filename>.sh
```

Add install commands:
```
#!/bin/sh
apt-get update ; apt-get install sudo ; sudo apt-get update ; sudo apt-get install git; sudo apt-get install nano
git clone https://github.com/matterport/Mask_RCNN.git
apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim unzip openssh-client wget \
    build-essential cmake \
    libopenblas-dev
apt-get install -y --no-install-recommends python3.5 python3.5-dev python3-pip python3-tk && \
    pip3 install pip==9.0.3 --upgrade && \
    pip3 install --no-cache-dir --upgrade setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases
apt-get install -y --no-install-recommends libjpeg-dev zlib1g-dev && \
    pip3 --no-cache-dir install Pillow
pip3 --no-cache-dir install \
    numpy scipy sklearn scikit-image==0.13.1 pandas matplotlib Cython requests pandas imgaug
pip3 --no-cache-dir install awscli --upgrade
pip3 --no-cache-dir install jupyter && \
    mkdir /root/.jupyter && \
    echo "c.NotebookApp.ip = '*'" \
         "\nc.NotebookApp.open_browser = False" \
         "\nc.NotebookApp.token = ''" \
         > /root/.jupyter/jupyter_notebook_config.py

```







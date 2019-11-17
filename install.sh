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
pip3 --no-cache-dir install tensorflow-gpu
apt-get install -y --no-install-recommends \
    libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgtk2.0-dev \
    liblapacke-dev checkinstall
pip3 install opencv-python
pip3 install --no-cache-dir --upgrade h5py pydot_ng keras
pip3 install --no-cache-dir git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI
pip3 install scikit-image==0.14.2
echo "deb http://us.archive.ubuntu.com/ubuntu/ yakkety universe" | sudo tee -a /etc/apt/sources.list
sudo apt-get update
sudo apt-get install libjasper-dev
sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt update
sudo apt install libjasper1 libjasper-dev 
apt-get install libgtk2.0-dev

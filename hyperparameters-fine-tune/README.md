# Fine tune hyperparameters in Mask RCNN model
There are six hyperparameters I changed in my dissertation: pre-trained net, activation function, learning rate, optimizer, Dropout function and data size.
## pre-trained net
To change a pre-trained net to from coco to imagenet, update new net when training:
```
cd /path/to/dissertation-master
python3 custom.py train --dataset=/path/to/datasetfolder --weights=imagenet
```
## activation function
To change activation function from "relu" to "LeakyRelu"
```
cd /path/to/model.py
nano model.py 
```
In model.py, edit every line
```
x = KL.Activation('relu')(x)
```
to 
```
x = KL.LeakyReLU(alpha=0.01)(x)
```
Then, press ctrl + o + Enter (save) and ctrl + x + Enter (exit)
## learning rate
```
cd /path/to/config.py
nano config.py
```
In config.py, update LEARNING_RATE
```
LEARNING_RATE = <set a new learning rate>
```

## optimizer

To change optimizer:
```
cd /path/to/config.py
nano config.py
```
In config.py, update optimizer 
```
OPTIMIZER = "ADAM"
```
or 
```
OPTIMIZER = "SDG"
```

## Dropout function
To add Dropout function:
```
cd /path/to/model.py
nano model.py
```
add 
```
x = KL.Dropout(0.5)(x)
```
after every Relu or leakyRelu function

## dataset size
simply create more synthetic images and annotations can help increase accuracy. 

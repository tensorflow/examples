## TensorFlow 2.0 implementation of Yolov3 
A minimal tensorflow implementation of YOLOv3, with support for training, inference and evaluation.

## Installations
install requirements and download pretrained weights
```
$ pip3 install -r requirements.txt
$ wget https://pjreddie.com/media/files/yolov3.weights
```

## Demo 
We will use pretrained weights to make predictions on image.


```
$ python detect.py
```
## Result
<img src="https://github.com/abidKiller/examples/blob/yolov3-example/tensorflow_examples/models/yolov3/data/result.jpg">

## Training
Download [yymnist](https://github.com/YunYang1994/yymnist) dataset and make data.

```
$ git clone https://github.com/YunYang1994/yymnist.git
$ python yymnist/make_data.py --images_num 1000 --images_path ./data/dataset/train --labels_txt ./data/dataset/yymnist_train.txt
$ python yymnist/make_data.py --images_num 200  --images_path ./data/dataset/test  --labels_txt ./data/dataset/yymnist_test.txt
```
Open `config.py` and do some configurations
```
__C.YOLO.CLASSES                = "./data/classes/yymnist.names"
```

Finally, you can train it and then evaluate your model

```
$ python train.py
$ tensorboard --logdir ./data/log
$ python test.py
$ cd ../mAP
$ python main.py        # Detection images are expected to save in `YOLOV3/data/detection`
```

## Useful Resources and Links

- [YOLO website](https://pjreddie.com/darknet/yolo/)
- [YOLOv3 paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- [NMS paper](https://arxiv.org/pdf/1704.04503.pdf)
- [NMS implementation](https://github.com/bharatsingh430/soft-nms)
- [GIOU Paper](https://giou.stanford.edu/GIoU.pdf)
- [DarkNet Implementation](https://github.com/pjreddie/darknet)
- [YOLO implementation](https://github.com/zzh8829/yolov3-tf2)






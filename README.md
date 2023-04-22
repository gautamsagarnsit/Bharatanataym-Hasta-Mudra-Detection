# YOLOv6

### Install

```shell
git clone https://github.com/meituan/YOLOv6
cd YOLOv6
pip install -r requirements.txt
```

### Inference

First, download a pretrained model from the YOLOv6 [release](https://github.com/meituan/YOLOv6/releases/tag/0.1.0)

Second, run inference with `tools/infer.py`

```shell
python tools/infer.py --weights yolov6s.pt --source img.jpg / imgdir
                                yolov6n.pt
```

### Training


```shell
python tools/train.py --batch 32 --conf configs/yolov6s.py --data data/coco.yaml --device 0
                                        configs/yolov6n.py
```


- conf: select config file to specify network/optimizer/hyperparameters
- data: prepare [COCO](http://cocodataset.org) dataset, [YOLO format coco labes](https://github.com/meituan/YOLOv6/releases/download/0.1.0/coco2017labels.zip) and specify dataset paths in data.yaml
- make sure your dataset structure as fellows:
```
├── coco
│   ├── annotations
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   ├── images
│   │   ├── train2017
│   │   └── val2017
│   ├── labels
│   │   ├── train2017
│   │   ├── val2017
│   ├── LICENSE
│   ├── README.txt
```


### Evaluation

Reproduce mAP on COCO val2017 dataset

```shell
python tools/eval.py --data data/coco.yaml --batch 32 --weights yolov6s.pt --task val
                                                                yolov6n.pt
```

### Resume
If your training process is corrupted, you can resume training by
```
# single GPU traning.
python tools/train.py --resume
```
Your can also specify a checkpoint path to `--resume` parameter by
```
# remember replace /path/to/your/checkpoint/path to the checkpoint path which you want to resume training.
--resume /path/to/your/checkpoint/path

```
#ResNet18
## Step 1: Data Preparation
Data folder should be structured as follows:
├── Data/
│   ├── bht_mudra/
│   │   ├── image1.jpg
│   │   └── image2.jpg
|   ├── test_set/
|   ├── train.csv
|   ├── test.csv

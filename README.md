# AdaIN
PyTorch implementation of [this paper](https://arxiv.org/abs/1703.06868).
Original implementation can be found at [here](https://github.com/xunhuang1995/AdaIN-style).

## Clone
via ssh
```
$ git clone git@github.com:kukosmos/AdaIN-pytorch-2019.git
```
via https
```
$ git clone https://github.com/kukosmos/AdaIN-pytorch-2019.git
```

## Requirments
Install following applications
* python3.6+
* unzip

Install following python libraries
* Pillow
* [pytorch](https://pytorch.org)
* [torchvision](https://pytorch.org)
* tensorboard
* tensorboardX
* tqdm

## Train
To train a model, do the following steps.

### Prepare dataset
First, download the dataset with given script.
For example, to download ```coco2017train``` dataset, type in the following command:
```
$ ./download.sh coco2017train
```
Detailed usage of ```download.sh``` script can be found as fallows:
```
$ ./download.sh --help
```

### Train model
Train the model with ```train.py```.
For example, to train model with ```coco2017train``` as content and ```wikiart``` as style, type in the following command:
```
$ python train.py --content-dir data/coco2017train --style-dir data/wikiart
```

To check the logs that saved in ```logs``` directory, type in the following command:
```
$ tensorboard --logdir logs
```

### For more
Advanced options can be found with following command:
```
$ python train.py --help
```

## Generate Results
To generate sytled images, use ```test.py```.
For example, to generate a styled image from trained model ```models/adain.pth``` with ```images/content.jpg``` as a content image
and ```images/style.jpg``` as a style image, type in the following command:
```
$ python test.py --model models/adain.pth --content images/content.jpg --style images/style.jpg
```

### Interpolation
To mix the two or more styles in one content image, specify the interpolation weights as follows:
```
$ python test.py --model models/adain.pth --content images/content.jpg --style images/style1.jpg images/style2.jpg images/style3.jpg --interpolation-weights 2 3 4
```
Note that all style images should have weights.
In other words, the number of images should be equal to the number of weights.

### For more
More options can be found with following command:
```
$ python test.py --help
```

## References
* [1]: X. Huang and S. Belongie. "[Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)", in ICCV, 2017.
* [2]: [PyTorch implementation of AdaIN](https://github.com/naoto0804/pytorch-AdaIN)
* [3]: [COCO Dataset](http://cocodataset.org/#download)
* [4]: [Wikiart dataset](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset)

# 03_PlayWithGANs

This is an increment of Pix2Pix with cGAN and a combination of DragGAN and face-alignment.

## Requirements

To install requirements on Linux:

```bash
cd path/to/03_PlayWithGANs
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Deep Learning-Based Pix2Pix with cGAN

### Datasets

To download `facades` dataset, run this command on Linux:

```bash
bash cgan/download_facades_dataset.sh
```

To download `cityscapes` dataset, run this command on Linux:

```bash
sed -e 's/facades/cityscapes/g' cgan/download_facades_dataset.sh | bash
```

### Training

To train the model(s) in the paper, run this command:

```bash
python -u train.py > train.log
```

### Results

Validation after 400 epochs:

<img src="results/after400.png" alt="After 400 epochs" width="800">

Validation after 800 epochs:

<img src="results/after800.png" alt="After 800 epochs" width="800">

## DragGAN & Face-Alignment

### Run

```bash
python -u facegan/combination.py
```

### Results

Original image:

<img src="results/image_original.png" alt="Original" width="800">

Close eyes:

<img src="results/image_close_eyes.png" alt="Close eyes" width="800">

Expand eyes:

<img src="results/image_expand_eyes.png" alt="Expand eyes" width="800">

Close lips:

<img src="results/image_close_lips.png" alt="Close lips" width="800">

Smile mouth:

<img src="results/image_smile_mouth.png" alt="Smile mouth" width="800">

Slim face:

<img src="results/image_slim_face.png" alt="Slim face" width="800">

Dynamically changing videos are [here](results/). 

## Contributing

>📋 This repository is under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. Welcome to create [issues](https://github.com/iw16/DIP-Teaching/issues) and/or [PRs](https://github.com/iw16/DIP-Teaching/pulls). 

## Acknowledgements

- Thanks to the [bugs](https://github.com/opengvlab/draggan) by [OpenGVLab](https://github.com/opengvlab) where an implementation of DragGAN is hidden inside. 
- Thanks to [face-alignment](https://github.com/1adrianb/face-alignment) by [Adrian Bulat](https://github.com/1adrianb). 
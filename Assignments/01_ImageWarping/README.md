# Assignment 1 - Image Warping

### In this assignment, you will implement basic transformation and point-based deformation for images.

### Resources:
- [Teaching Slides](https://rec.ustc.edu.cn/share/afbf05a0-710c-11ef-80c6-518b4c8c0b96) 
- [Paper: Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf)
- [Paper: Image Warping by Radial Basis Functions](https://www.sci.utah.edu/~gerig/CS6640-F2010/Project3/Arad-1995.pdf)
- [OpenCV Geometric Transformations](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
- [Gradio: 一个好用的网页端交互GUI](https://www.gradio.app/)

### 1. Basic Image Geometric Transformation (Scale/Rotation/Translation).
Fill the [Missing Part](run_global_transform.py#L21) of 'run_global_transform.py'.


### 2. Point Based Image Deformation.

Implement MLS or RBF based image deformation in the [Missing Part](run_point_transform.py#L52) of 'run_point_transform.py'.

---
## 一个作业提交模板 (里面的结果也可参考)


## Implementation of Image Geometric Transformation

This repository is Yiqi Wang's implementation of Assignment_01 of DIP. 

<img src="results/composite.png" alt="composition of global warpings" width="800">

<img src="results/mls.png" alt="MLS warping" width="800">

## Requirements

To install requirements:

```setup
python -m pip install -r requirements.txt
```


## Running

To run basic transformation, run:

```basic
python run_global_transform.py
```

To run point guided transformation, run:

```point
python run_point_transform.py
```

## Results (need add more result images)
### Basic Transformation
<img src="results/scale-only.png" alt="scale only" width="800">
<img src="results/rotate-only.png" alt="rotate only" width="800">
<img src="results/move-x-only.png" alt="move x only" width="800">
<img src="results/flip-x-only.png" alt="flip x only" width="800">
<img src="results/composite.png" alt="composite" width="800">

### Point Guided Deformation:
<img src="results/mls.png" alt="MLS warping" width="800">

## Acknowledgement

>📋 Thanks for the algorithms proposed by [Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf).

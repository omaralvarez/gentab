<p align="center">
    <a>
	    <img src='https://img.shields.io/badge/python-3.10%2B-blueviolet' alt='Python' />
	</a>
    <a>
	    <img src='https://img.shields.io/badge/code%20style-black-black' />
	</a>
	<a href="">
  		<img src="https://colab.research.google.com/assets/colab-badge.svg"/>
	</a>
    <a href='https://opensource.org/license/gpl-3-0'>
	    <img src='https://img.shields.io/badge/license-GPLv3-blue' />
	</a>
</p>

# synthtab

Synthetic Tabular Data Generation Library

## Overview

This Python library specializes in the generation of synthetic tabular data. It has a diverse range of statistical, Machine Learning (ML) and Deep Learning (DL) methods to accurately capture patterns in real datasets and replicate them in a synthetic context. It has multiple applications including pre-processing of tabular datasets, data balancing, resampling...

## Features

:nut_and_bolt: Pre-process your data.

:clock130: State-of-the-art models.

:recycle: Easy to use and customize. 

## Install

The `synthtab` library is available using pip. We recommend using a virtual environment to avoid conflicts with other software on your machine.

``` bash
pip install synthtab
```

## Available Generators

Below is the list of the generators currently available in the library.

### Statistical

|               Model                  |                                                                                    Example                                                                                    |                     Paper                    |
|:--------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| Random Over-Sampling      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://link.springer.com/article/10.1007/s10618-012-0295-5)
| SMOTE                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |            [link](https://arxiv.org/abs/1106.1813)                                  |                                                                            |
| ADASYN      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://ieeexplore.ieee.org/document/4633969)
| Gaussian Copula      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://ieeexplore.ieee.org/abstract/document/7796926)

### GAN

|               Model                  |                                                                                    Example                                                                                    |                     Paper                    |
|:--------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| TVAE      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://arxiv.org/abs/1907.00503)
| CTGAN      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://arxiv.org/abs/1907.00503)
| CTAB-GAN      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://proceedings.mlr.press/v157/zhao21a.html)
| CTAB-GAN+      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |  [link](https://arxiv.org/abs/2204.00401)

### Diffusion

|               Model                  |                                                                                    Example                                                                                    |                     Paper                    |
|:--------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| ForestDiffusion      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://arxiv.org/abs/2309.09968)
| AutoDiffusion      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://arxiv.org/abs/2310.15479)


### LLM

|               Model                  |                                                                                    Example                                                                                    |                     Paper                    |
|:--------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| GReaT      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://arxiv.org/abs/2210.06280)
| Tabula      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://arxiv.org/abs/2310.12746)

### Hybrid

|               Model                  |                                                                                    Example                                                                                    |                     Papers                    |
|:--------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| Copula GAN      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://ieeexplore.ieee.org/abstract/document/7796926) [link](https://arxiv.org/abs/1907.00503)

## Example

``` python
from synthtab.generators import AutoDiffusion
from synthtab.data import Config, Dataset
from synthtab.console import console

from sklearn import svm

config = Config("datasets/playnet/info.json")

dataset = Dataset(config)
dataset.reduce_size({
    "left_attack": 0.65,
    "right_attack": 0.65,
    "right_transition": 0.65,
    "left_transition": 0.65,
    "time_out": 0.65,
})
dataset.reduce_mem()

generator = AutoDiffusion(dataset)
generator.generate()
generator.evaluate(svm.SVM(gamma=0.001, C=100.0))
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts())
```

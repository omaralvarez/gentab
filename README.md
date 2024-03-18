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

# GenTab

Synthetic Tabular Data Generation Library

## Overview

This Python library specializes in the generation of synthetic tabular data. It has a diverse range of statistical, Machine Learning (ML) and Deep Learning (DL) methods to accurately capture patterns in real datasets and replicate them in a synthetic context. It has multiple applications including pre-processing of tabular datasets, data balancing, resampling...

## Features

:nut_and_bolt: Pre-process your data.

:clock130: State-of-the-art models.

:recycle: Easy to use and customize. 

## Install

The `gentab` library is available using pip. We recommend using a virtual environment to avoid conflicts with other software on your machine.

``` bash
pip install gentab
```

## Available Generators

Below is the list of the generators currently available in the library.

### Linear

|               Model                  |                                                                                    Example                                                                                    |                     Paper                    |
|:--------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| Random Over-Sampling      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://link.springer.com/article/10.1007/s10618-012-0295-5)
| SMOTE                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |            [link](https://arxiv.org/abs/1106.1813)                                  |                                                                            |
| ADASYN      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://ieeexplore.ieee.org/document/4633969)

### PDF
|               Model                  |                                                                                    Example                                                                                    |                     Paper                    |
|:--------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| Gaussian Copula      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://ieeexplore.ieee.org/abstract/document/7796926)


### AE

|               Model                  |                                                                                    Example                                                                                    |                     Paper                    |
|:--------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| TVAE      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://arxiv.org/abs/1907.00503)

### GAN

|               Model                  |                                                                                    Example                                                                                    |                     Paper                    |
|:--------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| CTGAN      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://arxiv.org/abs/1907.00503)
| CTAB-GAN      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://proceedings.mlr.press/v157/zhao21a.html)
| CTAB-GAN+      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |  [link](https://arxiv.org/abs/2204.00401)

### Diffusion

|               Model                  |                                                                                    Example                                                                                    |                     Paper                    |
|:--------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| ForestDiffusion      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://arxiv.org/abs/2309.09968)

### LLM

|               Model                  |                                                                                    Example                                                                                    |                     Paper                    |
|:--------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| GReaT      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://arxiv.org/abs/2210.06280)
| Tabula      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://arxiv.org/abs/2310.12746)

### Hybrid

|               Model                  |                                                                                    Example                                                                                    |                     Papers                    |
|:--------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| Copula GAN      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://ieeexplore.ieee.org/abstract/document/7796926) [link](https://arxiv.org/abs/1907.00503)
| AutoDiffusion      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://arxiv.org/abs/2310.15479)

## Examples

### Generation

``` python
from gentab.generators import AutoDiffusion
from gentab.evaluators import MLP
from gentab.data import Config, Dataset
from gentab.utils import console

config = Config("configs/playnet.json")

dataset = Dataset(config)
dataset.reduce_size(
    {
        "left_attack": 0.97,
        "right_attack": 0.97,
        "right_transition": 0.9,
        "left_transition": 0.9,
        "time_out": 0.8,
        "left_penal": 0.5,
        "right_penal": 0.5,
    }
)
dataset.merge_classes(
    {
        "attack": ["left_attack", "right_attack"],
        "transition": ["left_transition", "right_transition"],
        "penalty": ["left_penal", "right_penal"],
    }
)
dataset.reduce_mem()

console.print(dataset.class_counts(), dataset.row_count())
generator = AutoDiffusion(dataset)
generator.generate()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

evaluator = MLP(generator)
evaluator.evaluate()

dataset.save_to_disk(generator)
```

### Tuning

``` python
from gentab.generators import AutoDiffusion
from gentab.evaluators import LightGBM
from gentab.tuners import AutoDiffusionTuner
from gentab.data import Config, Dataset

config = Config("configs/adult.json")

dataset = Dataset(config)
dataset.merge_classes({
    "<=50K": ["<=50K."], ">50K": [">50K."]
})
dataset.reduce_mem()

generator = AutoDiffusion(dataset)

evaluator = LightGBM(generator)

trials = 10
time = 60 * 60 * 8
tuner = AutoDiffusionTuner(evaluator, trials, timeout=time)
tuner.tune()
tuner.save_to_disk()
```

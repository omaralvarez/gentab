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

This is a Python library designed for synthetic tabular data generation. It uses a variety of algorithms, ML and DL models to learn patterns from real data and emulate them synthetically. Its applications encompass tabular dataset pre-processing, balancing, resampling...

## Features

:nut_and_bolt: Pre-process your data.

:clock130: State-of-the-art models.

:recycle: Easy to use and customize. 

## Available Models

Below is the list of the models currently available in the library.

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
| AutoDiffusion      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://arxiv.org/abs/2310.15479)

### Hybrid

|               Model                  |                                                                                    Example                                                                                    |                     Papers                    |
|:--------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| Copula GAN      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | [link](https://ieeexplore.ieee.org/abstract/document/7796926) [link](https://arxiv.org/abs/1907.00503)

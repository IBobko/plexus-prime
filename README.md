# Project Name

## Introduction

(Your project description goes here)

## Installation and Setup

### Prerequisites

Ensure that you have [Anaconda](https://www.anaconda.com/products/distribution)
or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

### Initializing the 'mnist_img' directory

To create and initialize the 'mnist_img' directory, run the `save_mnist.py` script. This script will generate the '
mnist_img' directory (if it does not exist) and save the MNIST images to it. You can run the script by executing the
following command:

```bash
python save_mnist.py
```

### Creating an environment file

Before committing your project, it's good practice to generate an `environment.yml` file. This will allow others to
replicate your project's environment with ease.

You can do this by using the following command:

```bash
conda env export | grep -v "prefix: " > environment.yml
```

# *Pushing the Boundaries of Private, Large-Scale Query Answering*

This repository contains the full implementation of the Relaxed Adaptive Projection (RAP) mechanism used in the experiments of the *Pushing the Boundaries of Private, Large-Scale Query Answering* paper by Brendan Avent and Aleksandra Korolova.


## Setup Instructions

This project was developed to primarily to run on systems with Python 3.6+ that have an Nvidia GPU. It is assumed that the relevant CUDA and CuDNN libraries are installed and compatible with both the GPU and the (to-be-installed) JAX version.

Installing the Python packages required by this implementation is done by executing the pip command:
`pip install -r requirements.txt`

To answer very large sets of queries (> 2*10^9), JAX must be set to 64-bit mode on the underlying system as described [here](https://github.com/google/jax#current-gotchas). On Linux from a BASH shell, a simple way to set this permanently is to execute the following command:
`echo 'export JAX_ENABLE_X64=True' >> ~/.bashrc`

The datasets evaluated in the paper are the [ADULT](https://github.com/ryan112358/private-pgm/tree/master/data) and [LOANS](https://github.com/giusevtr/fem/tree/master/datasets) datasets.

These datasets can be automatically downloaded by running the `downloader.py` file in the `data` directory. Alternatively, the CSVs can be manually downloaded and placed in their corresponding subdirectories within the `data` directory.

Finally, ensure that the root of this project is in the system's PYTHONPATH.


## Implementation Information

This implementation is split between two primary directories at the project's root: `data` and `rap`.

The `data` directory contains files related to downloading, importing, and managing, and onehot encoding/decoding the relevant datasets.

The `rap` directory contains files for managing individual r-of-k thresholds, managing workloads of r-of-k thresholds, executing the RAP mechanism, and evaluating/storing the results of the RAP mechanism and the other simple baseline mechanisms.

`example.py` is a simple example script that illustrates how to programmatically set up and run a single simulation of the RAP mechanism.

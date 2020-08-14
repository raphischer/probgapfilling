# Spatio-temporal Gap Filling with Machine Learning - probgf

## General Information

This repository contains spatio-temporal gap filling software.
Our probabilistic gap filling approach scientifically published in October 2020, we here allow reviewers and readers to reproduce and better understand our results.

We deploy probabilistic machine learning (ML) methods on incomplete dataset to reconstruct missing values.
Keep in mind that this software serves as an exemplary implementation of theoretical methods and is intended for scientific usage.

## Software Requirements

The software is implemented and tested with Python 3.6, and uses some external Python packages.
The easiest way of use is to to create an Anaconda environment from the `environment.yml` file.
If you want to use a custom Python installation, you can check this file for required packages.
The repository also comes with a Dockerfile if you want to run the software in a Docker container (in the container make sure to first run `conda activate probgf`).
Before usage, you should install this package by running `pip install .` in the root directoriy of its repository (already done in Docker image).

## How To Use

For standalone gap filling, simply run `python -m probgf`. Pass `-h` for a detailed overview of command line parameters.
You can clean up the current directory of any `probgf` related files and results by passing `-d C` on the command line (the corresponding function is also available from `probgf.helpers`).

`scripts/gap_filling_viewer.py` allows to interactively inspect gap filling results, it is simply started via command line.
It uses *Pillow* and *Tkinter*, and can be controlled via mouse and keyboard.
Pass `-h` for a detailed overview of command line parameters.

## Data for Gap Filling

The implemented gap filling methods have been tested on synthetic data as well as real-world remote sensing datasets.
Via `-d Chess`, experiments can be run with the synthetic data and gaps at `t=2`.
By passing `-d Chess2`, experiments use the synthetic data where `t=2` is completely missing.

The `Dortmund From Space 2018` data (aka `GER`) can be found at <https://www.dropbox.com/sh/ohbb4zpae9djb3z/AADi5qGbsPB2peLGg2-gh8LWa>.
For usage, download it directly, unzip the obtained archive, and pass `-d [download path]/dortmund_from_space_2018/` to the software. More information on this data can be found in the corresponding *README*.

## Outline of Gap Filling Method

1. Load and reshape spatio-temporal data and gap masks

2. Set up a cross-validation (CV) and add artificial gaps in each split

3. For each CV split:

    a. Learn the model on the training data

    b. Predict the missing entries in the test data

    c. Evaluate results on artificial gaps

4. Merge reports and visualize results

## Attached Results

Images and error reports on `GER` data can be found at <https://www.dropbox.com/sh/dojhb0dhzljznyy/AAC-PVlGidGFkx-RFvQw5oG3a>, and results on `FRA` data can be downloaded from <https://www.dropbox.com/sh/rj959rhjr9ndec0/AAAOA8vSzv0pANMFZXstjxwWa>.
This software allows to reproduce the results, the command line arguments can be derived from filenames.
Resulting images can also be inspected with the `gap_filling_viewer`, by passing the downloaded directories as arguments. As an example, for viewing *CROSS* gap filling results on `GER` data, run:

`python scripts/gap_filling_viewer.py`
`-l [path]/imgs/original_outline/`
`-m [path]/imgs/mask/`
`-r [path]/imgs/pred_outline_mrf_s0_01_tp_tp_0_1_cross3_sup_em5_k32means_spatial_clouds0_2/`
`-R [path]/report_mrf_s0_01_tp_tp_0_1_cross3_sup_em5_k32means_spatial_clouds0_2.csv`
`-y 2018`

## Reference

If you use this code or the data, please link back to <https://github.com/raphischer/probgapfilling>.

## Term of Use

Please refer to `LICENSE.md` for terms of use.

Copyright (c) 2020 Raphael Fischer

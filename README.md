
# ForestCHAT

TODO - This repo will contain the implementation of an augmented TEOChat model for the CAM-ForestNet dataset

# Repository setup

Create a virtual environment using pyenv for python 3.8 via the following commands:

1. Install python 3.8.20 with: `pyenv install -v 3.8.20`
1. Create a virtual environment with: `pyenv virtualenv 3.8.20 forestchat`
1. Activate the environment: `pyenv activate forestchat`
1. Confirm the correct python version is selected with `python -V` - it should be 3.8.20
1. Select this virtual environment for any notebook kernels
1. Install pip requirements via `pip install -r requirements.txt`
    1. If you run into issues with this if needing to recreate the environment, try installing via `pip install --no-cache-dir -r requirements.txt` or `pip install --force-reinstall -r requirements.txt` to fix.

Citations:

@article{10.1117/1.JRS.17.036502,
author = {Bella Septina Ika Hartanti and Valentino Vito and Aniati Murni Arymurthy and Adila Alfa Krisnadhi and Andie Setiyoko},
title = {{Multimodal SuperCon: classifier for drivers of deforestation in Indonesia}},
volume = {17},
journal = {Journal of Applied Remote Sensing},
number = {3},
publisher = {SPIE},
pages = {036502},
keywords = {deforestation driver classification, contrastive learning, class imbalance, multimodal fusion, Machine learning, Education and training, Data modeling, Image fusion, Performance modeling, Atmospheric modeling, Data fusion, Deep learning, Landsat, RGB color model},
year = {2023},
doi = {10.1117/1.JRS.17.036502},
URL = {https://doi.org/10.1117/1.JRS.17.036502}
}


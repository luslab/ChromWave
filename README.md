# ChromWave - a deep learning model for sequential genomic data

Note that while you can use pretrained ChromWave models provided here to make transcription factor and nucleosome binding without
the use of a GPU, any training of new models will require a GPU.

## Installation
Note: These installation steps assume that you are on a Linux environment.

### Install Miniconda for python 3
We recommend to install ChromWave in a conda environment. 
If you are on Windows or a Mac, you will need to download the right Miniconda2 installation file and install conda and pip packages manually as described below.  We provide all [necessary files](https://github.com/luslab/GenomeNet/tree/master/conda%20installation%20files).

First clone ChromWave using `git`:

```sh
git clone https://github.com/luslab/ChromWave.git
```

If you are on linux or Mac you can use the provided miniconda install script in the `conda installation files` folder. In a terminal window `cd` into the ChromWave folder and install miniconda from the provided script by typing 
```sh
bash Miniconda3-latest-Linux-x86_64.sh
```
If you are on a MacOS use the provided Mac files, eg. 
```sh
bash Miniconda3-latest-MacOSX-x86_64
```

Once you have Miniconda3 installed, create the chromwave environment (still in the ChromWave folder)
```sh
conda env create -f environment_linux.yml
```
of if you are on MacOS type

```sh
conda env create -f environment_mac.yml
```

Alternatively, if you have your own version of conda installed, create a new environment using python3.7and then install all dependencies in conda: 

``` sh
conda create -n chromwave python=3.7
conda install -c anaconda tensorflow 
```
of if you have access to a GPU
``` sh
conda install -c anaconda tensorflow-gpu
```
```sh
conda install keras pandas --channel conda-forge
conda install scikit-learn matplotlib biopython pyyaml h5py --channel conda-forge
conda install scikit-image joblib hyperas  --channel conda-forge
```
followed by the pip dependencies: 
```sh
pip install roman
```

You may also consider installing the following **optional dependencies** :

- [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) (recommended if you plan on running Keras on GPU).
- HDF5 and [h5py](http://docs.h5py.org/en/latest/build.html) (required if you plan on saving Keras models to disk). H5py is included in the provided conda environments.
- [graphviz](https://graphviz.gitlab.io/download/) and [pydot_ng](https://pypi.org/project/pydot-ng/) (used by [visualization utilities](https://keras.io/visualization/) to plot model graphs). Pydot_ng is included in the provided conda environments and enabeles visualisation of the model architectures.
- [jupyter-lab](https://jupyter.org/install) via conda: ```conda install -c conda-forge jupyterlab```
- [hyperopt](http://hyperopt.github.io/hyperopt/) and [hyperas](https://maxpumperla.com/hyperas/) to run hyperoptimisations as in the [examples provided](scripts/Hyperoptimisations) (both are provided in the conda environments)

### Install ChromWave

Now you can install ChromWave:

First, activate the chromwave environment (`source activate chromwave`), `cd` into the ChromWave folder and run the install command:
```sh
cd ChromWave
python setup.py install
```
If you plan to continue developing the package and want to make local changes that are reflected in your installation, run instead

```sh

python setup.py develop
```


## Quick start

We have assembled some generic workflows as tutorials as [jupyter notebooks](scripts/Tutorials_Workflows).
This includes a quick start in using the models and how to load and preprocess data (Tutorials 1&2) as well as
other workflows showing how to produce most of the figures in our publication 'ChromWave: Deciphering the DNA-encoded competition between DNA-binding factors with deep neural networks'.

We also provide scripts that can be run in the command line for the [optimisation of the hyperparameters with hyperopt]( scripts/Hyperoptimisations).

## The models
In the [model directory](models), we are providing all model files to be able to load the models again to run them on your own
genomic data to produce DNA-binding prediction profiles. The model directories each also contain a prediction folder
with bigWig files of the predictions for transcription factor and nucleosome binding in the yeast genome([TF-Nucleosome model](models/tf-nucleosome/sacCer3/predictions)), nucleosome occupancy in the yeast genome [*in vitro*](models/nucleosomes/sacCer1_inVitro/predictions) and [*in vivo*](models/nucleosomes/sacCer1_inVivo/predictions) and  finally [nucleosome occupancy](models/nucleosomes/hg38_promoter/predictions) predictions +/-1000bp around the annotated transcriptional start sites in hg38. For more information on these models see our publication 'ChromWave: Deciphering the DNA-encoded competition between DNA-binding factors with deep neural networks' and the provided [jupyter notebooks](scripts/Tutorials_Workflows). 

## The data
The [data directory](data) contains all data necessary to train the provided models. This includes the genomic data and TSS annotation in the case of the human promoter model, as well as all binding profile data. ChromWave can readily be trained from csv files with binding occupancies, [computeMatrix](https://deeptools.readthedocs.io/en/develop/content/tools/computeMatrix.html?highlight=computeMatrix) output of [deeptools](https://deeptools.readthedocs.io/en/develop/) as well as bed file. If you are planning of using bed files, please make sure that you have no ranges of width>1 e.g. you must have a score at each base-pair position (missig values will be imputed with the genomic average). We are providing an example script of how to do this in R in the [data directory](data).


## R integration
Our models can easily be loaded in R to predict DNA-binding profiles in the genome of choice using the packages `reticulate` and `KerasR`. We have provided some examples on how to do this as [R scripts](https://github.com/luslab/chromWaveR).


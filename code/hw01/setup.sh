#!/usr/bin/env bash

wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod 755 miniconda.sh
./miniconda.sh -b
export PATH=$HOME/miniconda3/bin:$PATH
conda config --append channels conda-forge 
conda update --yes conda
conda install python=3.8

conda install --yes numpy scipy scikit-learn pip cython nltk matplotlib
which python
which pip
pip install -r /autograder/source/requirements.txt
pip install microtc
pip install b4msa
pip install evomsa
pip install text_models
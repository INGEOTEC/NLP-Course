#!/usr/bin/env bash

wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod 755 miniconda.sh
./miniconda.sh -b
export PATH=$HOME/miniconda3/bin:$PATH
conda config --append channels conda-forge
conda update --yes conda

apt-get install -y python3 python3-pip python3-dev
# pip3 install numpy
# pip3 install cython

conda install --yes numpy scipy scikit-learn spacy
which python
which pip3
pip3 install -r /autograder/source/requirements.txt

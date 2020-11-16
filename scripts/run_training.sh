#!/bin/bash -v
#PBS -q gpu
#PBS -N versetorch
#PBS -l select=1:ncpus=2:ngpus=2:mem=10gb:scratch_local=10gb:cl_adan=True
#PBS -l walltime=0:10:00 
#PBS -j oe

module add python-3.6.2-gcc
module add python36-modules-gcc
module add cuda-10.0
module add cudnn-7.0

cd $PBS_O_WORKDIR

source scripts/venv.sh
export PYTHONPATH=/storage/praha1/home/memduh/versetorch/venv/
export PYTHON=/storage/praha1/home/memduh/versetorch/venv/bin/python
$PYTHON src/train.py

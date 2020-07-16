#!/bin/bash -v
#PBS -q gpu
#PBS -N versetorch
#PBS -l select=1:ncpus=1:ngpus=1:mem=10gb:scratch_local=10gb:cl_adan=True
#PBS -l walltime=0:30:00 
#PBS -j oe

module add python-3.6.2-gcc
module add cuda-10.0
module add cudnn-7.0

cd $PBS_O_WORKDIR

source scripts/venv.sh
export PYTHONPATH=/storage/plzen1/home/memduh/versetorch/venv/
python src/train.py

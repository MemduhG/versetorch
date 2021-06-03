#PBS -l select=1:ncpus=4:mem=10gb:scratch_local=10gb
#PBS -j oe

module add python-3.6.2-gcc
module add python36-modules-gcc
module add cuda-10.0
module add cudnn-7.0

cd $PBS_O_WORKDIR

rm -rf runs
source scripts/venv.sh
export PYTHONPATH=/storage/plzen1/home/memduh/versetorch/venv/
export PYTHON=/storage/plzen1/home/memduh/versetorch/venv/bin/python
$PYTHON -m pip install tensorboard TurkishStemmer pronouncing
$PYTHON src/utils/tb.py


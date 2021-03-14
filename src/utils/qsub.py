import sys
import subprocess
src_files = {"cz-baseline": "data/cz/cz.dev.src",
             "eng-baseline": "data/en/eng.dev.src",
             "tur-baseline": "data/tr/tur.dev.src"}

languages = {"cz-baseline": "cz",
             "eng-baseline": "en",
             "tur-baseline": "tr"}


template_script = """#PBS -q gpu
#PBS -l select=1:ncpus=1:ngpus=1:mem=10gb:scratch_local=10gb:cl_adan=True
#PBS -l walltime=4:00:00 
#PBS -j oe

module add python-3.6.2-gcc
module add python36-modules-gcc
module add cuda-10.0
module add cudnn-7.0

cd $PBS_O_WORKDIR

source scripts/venv.sh
export PYTHONPATH=/storage/plzen1/home/memduh/versetorch/venv/
export PYTHON=/storage/plzen1/home/memduh/versetorch/venv/bin/python
$PYTHON src/utils/translate.py \
--language {language} --max_len 256 --checkpoint {checkpoint} --output {output} --input {input}"""


def qsub(save_file, steps):
    _, experiment, _ = save_file.split("/")
    input = src_files[experiment]
    checkpoint = save_file
    output = "translations/{experiment}/{steps}".format(experiment=experiment, steps=steps)
    new_script = template_script.format(input=input, checkpoint=checkpoint, output=output,
                                        language=languages[experiment])
    script_path = "/storage/plzen1/home/memduh/.scratch/{}-{}.sh".format(experiment, steps)
    with open(script_path, "w", encoding="utf-8") as outfile:
        outfile.write(new_script)
    subprocess.run(args=["qsub", script_path])

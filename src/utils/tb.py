import torch
from torch.utils.tensorboard import SummaryWriter
import os
import sacrebleu
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.rhyme import concurrent_score

ref_files = {"cz-baseline": "data/cz/cz.dev.tgt",
             "eng-baseline": "data/en/eng.dev.tgt",
             "tur-baseline": "data/tr/tur.dev.tgt"}

languages = {"cz-baseline": "cz", "eng-baseline": "en", "tur-baseline": "tr"}

references = {}

for exp in ref_files:
    ref = []
    if exp not in references:
        with open(ref_files[exp], "r", encoding="utf-8") as infile:
            for line in infile:
                ref.append(line.strip())
        references[exp] = ref[:]

writer = SummaryWriter()

experiments = os.listdir("translations")

for experiment in experiments:
    ref = references[experiment]
    exp_path = os.path.join("translations", experiment)
    for translation in sorted(os.listdir(exp_path), key=lambda x: int(x)):
        steps = int(translation)
        system_output = []
        file_path = os.path.join(exp_path, translation)
        with open(file_path, "r", encoding="utf-8") as infile:
            for line in infile:
                system_output.append(line.strip())
            bleu = sacrebleu.corpus_bleu(system_output, [ref])
            rhyme_score = concurrent_score(system_output, languages[experiment])
            print(experiment, translation, bleu.score, rhyme_score)

        wall = os.stat(file_path).st_mtime
        writer.add_scalar(experiment + "/BLEU", bleu.score, global_step=steps, walltime=wall)
        writer.add_scalar(experiment + "/Rhyme", rhyme_score, global_step=steps, walltime=wall)


writer.flush()
writer.close()

import torch
from torch.utils.tensorboard import SummaryWriter
import os
import sacrebleu

ref_files = {"cz-baseline": "data/cz/cz.dev.tgt",
             "eng-baseline": "data/en/eng.dev.tgt",
             "tur-baseline": "data/tr/tur.dev.tgt"}

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
        with open(os.path.join(exp_path, translation), "r", encoding="utf-8") as infile:
            for line in infile:
                system_output.append(line.strip())
            bleu = sacrebleu.corpus_bleu(system_output, [ref])
            print("val/" + experiment, translation, bleu.score)

        writer.add_scalar(experiment, bleu.score, steps)


writer.flush()
writer.close()
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import sacrebleu
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.rhyme import concurrent_score, score_originality, get_verse_ends

ref_files = {"cz-baseline": "data/cz/cz.dev.tgt",
             "eng-baseline": "data/en/eng.dev.tgt",
             "tur-baseline": "data/tr/tur.dev.tgt"}

src_files = {"cz-baseline": "data/cz/cz.dev.src",
             "eng-baseline": "data/en/eng.dev.src",
             "tur-baseline": "data/tr/tur.dev.src"}

languages = {"cz-baseline": "cz", "eng-baseline": "en", "tur-baseline": "tr"}

references = {}
sources = dict()

for exp in ref_files:
    ref = []
    if exp not in references:
        with open(ref_files[exp], "r", encoding="utf-8") as infile:
            for line in infile:
                ref.append(line.strip())
        references[exp] = ref[:]

for exp in src_files:
    source = []
    if exp not in sources:
        with open(src_files[exp], "r", encoding="utf-8") as infile:
            for line in infile:
                source.append(line.strip())
        sources[exp] = source[:]

writer = SummaryWriter()

experiments = os.listdir("translations")

for experiment in experiments:
    ref = references[experiment]
    src = sources[experiment]
    exp_path = os.path.join("translations", experiment)
    redif = "tur" in experiment

    num_lines = len(ref)
    for translation in sorted(os.listdir(exp_path), key=lambda x: int(x)):
        steps = int(translation)
        system_output = []
        file_path = os.path.join(exp_path, translation)
        all_copied, all_recon = 0., 0.
        with open(file_path, "r", encoding="utf-8") as infile:
            system_output = [x.strip() for x in infile.readlines()]
            bleu = sacrebleu.corpus_bleu(system_output, [ref])
            rhyme_score, copied, reconstructed = concurrent_score(system_output,
                                                                  languages[experiment],
                                                                  ref, src)
            print(experiment, translation, bleu.score, rhyme_score, copied, reconstructed)

        wall = os.stat(file_path).st_mtime
        writer.add_scalar(experiment + "/BLEU", bleu.score, global_step=steps, walltime=wall)
        writer.add_scalar(experiment + "/Rhyme", rhyme_score, global_step=steps, walltime=wall)
        writer.add_scalar(experiment + "/Copied", copied, global_step=steps, walltime=wall)
        writer.add_scalar(experiment + "/Reconstructed", reconstructed, global_step=steps, walltime=wall)

writer.flush()
writer.close()

import torch
from torch.utils.tensorboard import SummaryWriter
import os
import sacrebleu
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.rhyme import concurrent_score, score_originality, get_verse_ends, score_prose_translation

languages = {"cz-baseline": "cz", "cz-acc": "cz", "cz-dae": "cz",
                "eng-baseline": "en", "eng-acc": "en", "eng-dae": "en", 
                "tur-baseline": "tr", "tur-dae": "tr", "tur-acc": "tr"}

def get_files():
    ref_files = {"cz-baseline": "data/cz/cz.dev.tgt",
                "cz-acc": "data/cz/cz.dev.tgt",
                "cz-dae": "data/cz/cz.dev.tgt",
                "eng-baseline": "data/en/eng.dev.tgt",
                "eng-acc": "data/en/eng.dev.tgt",
                "eng-dae": "data/en/eng.dev.tgt",
                "tur-baseline": "data/tr/tur.dev.tgt",
                "tur-dae": "data/tr/tur.dev.tgt",
                "tur-acc": "data/tr/tur.dev.tgt"}

    src_files = {"cz-baseline": "data/cz/cz.dev.src",
                "cz-acc": "data/cz/cz.dev.src",
                "cz-dae": "data/cz/cz.dev.src",
                "eng-baseline": "data/en/eng.dev.src",
                "eng-acc": "data/en/eng.dev.src",
                "eng-dae": "data/en/eng.dev.src",
                "tur-baseline": "data/tr/tur.dev.src",
                "tur-dae": "data/tr/tur.dev.src",
                "tur-acc": "data/tr/tur.dev.src"}

    prose_files = {"cz-baseline": "data/cz/prose.txt",
                "cz-acc": "data/cz/prose.txt",
                "cz-dae": "data/cz/prose.txt",
                "eng-baseline": "data/en/prose.txt",
                "eng-dae": "data/en/prose.txt",
                "eng-acc": "data/en/prose.txt",
                "tur-baseline": "data/tr/prose.txt",
                "tur-dae": "data/tr/prose.txt",
                "tur-acc": "data/tr/prose.txt"}

    languages = {"cz-baseline": "cz", "cz-acc": "cz", "cz-dae": "cz",
                "eng-baseline": "en", "eng-acc": "en", "eng-dae": "en", 
                "tur-baseline": "tr", "tur-dae": "tr", "tur-acc": "tr"}

    references = {}
    sources = dict()
    prose_sources = dict()

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

    for exp in prose_files:
        source = []
        if exp not in prose_sources:
            with open(prose_files[exp], "r", encoding="utf-8") as infile:
                for line in infile:
                    source.append(line.strip())
            prose_sources[exp] = source[:]

    return references, sources, prose_sources

references, sources, prose_sources = get_files()



experiments = os.listdir("translations")

def write_evals(writer, experiment, translation, file_path, ref, src):
    writer = SummaryWriter("runs/{}-{}".format(experiment, translation))

    steps = int(translation)
    output_path = "translations/{}/{}".format(experiment, translation)
    with open(output_path, "r", encoding="utf-8") as infile:
        system_output = [x.strip() for x in infile.readlines()]
        bleu = sacrebleu.corpus_bleu(system_output, [ref])
        chrf = sacrebleu.corpus_chrf(system_output, [ref])
        rhyme_score, copied, reconstructed = concurrent_score(system_output,
                                                                languages[experiment],
                                                                ref, src)
        print(experiment, translation, bleu.score, rhyme_score, copied, reconstructed)

    wall = os.stat(file_path).st_mtime
    writer.add_scalar(experiment + "/CHRF", chrf.score, global_step=steps, walltime=wall)
    writer.add_scalar(experiment + "/BLEU", bleu.score, global_step=steps, walltime=wall)
    writer.add_scalar(experiment + "/Rhyme", rhyme_score, global_step=steps, walltime=wall)
    writer.add_scalar(experiment + "/Copied", copied, global_step=steps, walltime=wall)
    writer.add_scalar(experiment + "/Reconstructed", reconstructed, global_step=steps, walltime=wall)
    writer.flush()

def write_prose_evals(writer, experiment, translation, file_path, prose_src):
    writer = SummaryWriter("runs/{}-{}-prose".format(experiment, translation))

    steps = int(translation)
    output_path = "prose_translations/{}/{}".format(experiment, translation)
    with open(output_path, "r", encoding="utf-8") as infile:
        system_output = [x.strip() for x in infile.readlines()]
        prose_rhyme, prose_copied = score_prose_translation(system_output,
                                                            languages[experiment], prose_src)

    wall = os.stat(file_path).st_mtime
    writer.add_scalar(experiment + "/Prose-Rhyme", prose_rhyme, global_step=steps, walltime=wall)
    writer.add_scalar(experiment + "/Prose-Copied", prose_copied, global_step=steps, walltime=wall)
    writer.flush()


def write_all():
    exps = [x for x in experiments if "dae" not in x and "acc" not in x and "cz" in x]
    writer = SummaryWriter()
    for experiment in exps:
        ref = references[experiment]
        src = sources[experiment]
        prose_src = prose_sources[experiment]
        exp_path = os.path.join("translations", experiment)
        redif = "tur" in experiment

        num_lines = len(ref)
        
        for translation in sorted(os.listdir(exp_path), key=lambda x: int(x)):
            file_path = "checkpoints/{}/{}.pt".format(experiment, translation)
            write_evals(writer, experiment, translation, file_path, ref, src)

        prose_path = os.path.join("prose_translations", experiment)

        for translation in sorted(os.listdir(prose_path), key=lambda x: int(x)):
            write_prose_evals(writer, experiment, translation, file_path, prose_src)


    writer.flush()
    writer.close()


if __name__ == "__main__":
    write_all()
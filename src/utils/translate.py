import argparse
import os

import torch

from data_utils.batch import rebatch
from data_utils.data import get_training_iterators
from model.model import make_model, greedy_decode
from utils.utils import dataset_to_tok, get_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def translate_devset(checkpoint_path, dataset, vocab_size=32000, config=None):
    if config is None:
        experiment_name = dataset + "-baseline"
    else:
        experiment_name = dataset + "-" + config
    save_to = "translations/{}/{}".format(experiment_name, checkpoint_path.split("/")[-1].split(".")[0])
    model = make_model(vocab_size, vocab_size, N=6).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    tokenizer = get_tokenizer(dataset_to_tok[dataset])
    train, val, test = get_training_iterators(dataset)
    pad_idx = 3
    val_iter = (rebatch(pad_idx, b) for b in val)
    decoded = []
    for batch in val_iter:
        out = greedy_decode(model, batch.src, batch.src_mask, max_len=256, start_symbol=1)
        for row in out:
            to_decode = []
            for i in range(1, row.shape[0]):
                sym = row[i].item()
                if sym == 2:
                    break
                to_decode.append(int(sym))
            trans = tokenizer.DecodeIdsWithCheck(to_decode)
            print("Decoded: {}".format(trans).encode('utf-8'))
            decoded.append(trans)
    reordered = decoded[:]
    line_numbers = val.indices
    for c, sentence in enumerate(decoded):
        reordered[line_numbers[c]] = sentence
    if not os.path.exists("translations"):
        os.makedirs("translations")
    if not os.path.exists("translations/{}".format(experiment_name)):
        os.makedirs("translations/{}".format(experiment_name))
    with open(save_to, "w", encoding="utf-8") as outfile:
        for line in reordered:
            outfile.writelines(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=False)
    args = parser.parse_args()
    translate_devset(args.checkpoint, args.dataset, config=args.config)

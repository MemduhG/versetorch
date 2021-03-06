import argparse
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data_utils.batch import rebatch_single
from src.data_utils.data import make_val_iterator
from src.model.model import make_model, greedy_decode
from src.utils.utils import get_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def translate_devset(args):
    save_to = args.output
    model = make_model(32000, 32000, N=6).to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    tokenizer = get_tokenizer(args.language)
    val_iter, val_indices = make_val_iterator(args.input, tokenizer)
    pad_idx = 3
    val_iter = (rebatch_single(pad_idx, b) for b in val_iter)
    decoded = [""] * len(val_indices)
    for batch in val_iter:
        out = greedy_decode(model, batch.src, batch.src_mask, max_len=args.max_len, start_symbol=1)
        for c, decoded_row in enumerate(out):
            src_seq = [x for x in batch.src[:, c].tolist() if x != 3]
            index = val_indices[tuple(src_seq)]
            decoded[index] = decoded_row

    with open(save_to, "w", encoding="utf-8") as outfile:
        for line in decoded:
            outfile.writelines(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_len", type=int, required=True)
    args = parser.parse_args()
    translate_devset(args)

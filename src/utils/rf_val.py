import torch
import sys
import os
import argparse
from reformer_pytorch import ReformerEncDec

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.adafactor import Adafactor
from src.utils.utils import get_tokenizer
from src.model.model import NoamOpt
from src.data_utils.data import get_training_iterators, make_val_iterator
from src.utils.save import save_checkpoint, load_latest, load_checkpoint


save_every = 1800
save_path = "checkpoints/tur-rf"
MAX_SEQ_LEN = 1024

device = 0 if torch.cuda.device_count() > 0 else "cpu"

enc_dec = ReformerEncDec(dim=512, enc_num_tokens=32000, enc_depth=6, enc_max_seq_len=MAX_SEQ_LEN, dec_num_tokens=32000,
                         dec_depth=6, dec_max_seq_len=MAX_SEQ_LEN, ignore_index=3, pad_value=3).to(device)

optim = Adafactor(enc_dec.parameters())


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--ckpt")
arg_parser.add_argument("--input")
arg_parser.add_argument("--output")
args = arg_parser.parse_args()

#load_checkpoint(args.ckpt, enc_dec, optim)
tokenizer = get_tokenizer("tr")
val_iterator, val_indices = make_val_iterator(args.input, tokenizer)

decoded = [""] * len(val_indices) 

for batch in val_iterator:
    # TODO get a list of indices first!
    src = torch.transpose(batch.src, 0, 1).to(device)
    tgt = torch.ones((src.shape[0], 1), dtype=torch.long).to(device)
    generated = enc_dec.generate(src, tgt, seq_len=5, eos_token=2)
    for c, decoded_row in enumerate(generated):
        src_seq = [x for x in batch.src[:, c].tolist() if x != 3]
        index = val_indices[tuple(src_seq)]
        decoded[index] = decoded_row


with open(args.output, "w", encoding="utf-8") as outfile:
    for line in decoded:
        txt = tokenizer.Decode(line.tolist())
        outfile.write(txt + "\n")

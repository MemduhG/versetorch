import torch
from torch import nn
import sys, os
import sentencepiece as spm


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.utils import get_tokenizer


from src.data_utils.data import get_training_iterators

model = nn.Transformer(nhead=8, num_encoder_layers=6)
gen = nn.Linear(512, 32000)
embed = nn.Embedding(32000, 512)

crit = nn.CrossEntropyLoss()

tokenizer = get_tokenizer("tr")

with open("data/tr/tur.train.src") as infile:
    src_raw = [tokenizer.EncodeAsIds(infile.readline().strip()) for x in range(10)]

with open("data/tr/tur.train.tgt") as infile:
    tgt_raw = [tokenizer.EncodeAsIds(infile.readline().strip()) for x in range(10)]


for i in range(50):
    src, tgt = embed(src_raw), embed(tgt_raw)
    memory = model.encoder(src)
    out = model.decoder(tgt[:-1, :, :], memory, model.generate_square_subsequent_mask(tgt[:-1, :, :].shape[0]))

    pred = gen(out)

    loss = crit(pred.contiguous().view(-1, pred.size(-1)), tgt_raw[1:, :].contiguous().view(-1))
    print(loss)
    loss.backward()
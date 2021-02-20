import torch
import sys
import os
from reformer_pytorch import ReformerEncDec
from src.utils.utils import get_tokenizer
from src.model.model import NoamOpt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data_utils.data import get_training_iterators


MAX_SEQ_LEN = 1024

enc_dec = ReformerEncDec(dim=512, enc_num_tokens=32000, enc_depth=6, enc_max_seq_len=MAX_SEQ_LEN, dec_num_tokens=32000,
                         dec_depth=6, dec_max_seq_len=MAX_SEQ_LEN, ignore_index=3, pad_value=3).cuda()

opt = NoamOpt(512, 1, 2000, torch.optim.Adam(enc_dec.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

tokenizer = get_tokenizer("tr")
train_iter, valid_iter, test_iter, train_idx, dev_idx, test_idx = get_training_iterators("tur")

for batch in train_iter:
    src, tgt = torch.transpose(batch.src, 0, 1).cuda(), torch.transpose(batch.trg, 0, 1).cuda()
    input_mask = src != 3
    loss = enc_dec(src, tgt, return_loss=True, enc_input_mask=input_mask)
    print(loss)
    loss.backward()
    opt.step()

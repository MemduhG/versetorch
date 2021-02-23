import torch
import sys
import os
import time
from reformer_pytorch import ReformerEncDec

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.utils import get_tokenizer
from src.model.model import NoamOpt
from src.data_utils.data import get_training_iterators
from src.utils.save import save_checkpoint, load_latest

save_every=1800
save_path="checkpoints/tur-rf"
MAX_SEQ_LEN = 1024

enc_dec = ReformerEncDec(dim=512, enc_num_tokens=32000, enc_depth=6, enc_max_seq_len=MAX_SEQ_LEN, dec_num_tokens=32000,
                         dec_depth=6, dec_max_seq_len=MAX_SEQ_LEN, ignore_index=3, pad_value=3).cuda()

opt = NoamOpt(512, 1, 2000, torch.optim.Adam(enc_dec.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

tokenizer = get_tokenizer("tr")
train_iter, valid_iter, test_iter, train_idx, dev_idx, test_idx = get_training_iterators("tur")

steps = load_latest(save_path, enc_dec, opt)

last_saved = time.time()

for batch in train_iter:
    src, tgt = torch.transpose(batch.src, 0, 1).cuda(), torch.transpose(batch.trg, 0, 1).cuda()
    input_mask = src != 3
    try:
        loss = enc_dec(src, tgt, return_loss=True, enc_input_mask=input_mask)
    except AssertionError:
        print("Skipped overlong sample", src.shape, tgt.shape)
        continue
    print(loss, src.shape, tgt.shape)
    loss.backward()
    opt.step()
    opt.optimizer.zero_grad()
    steps += 1
    if time.time() - last_saved > save_every:
        print("Saving checkpoint at", steps, "steps with loss of", float(loss))
        save_checkpoint(enc_dec, opt, steps, save_path)
        last_saved = time.time()

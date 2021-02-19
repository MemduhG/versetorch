import torch
from torch import nn
import sys, os
import sentencepiece as spm
from reformer_pytorch import ReformerLM

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.utils import get_tokenizer


from src.data_utils.data import get_training_iterators

model = nn.Transformer(nhead=8, num_encoder_layers=6)
gen = nn.Linear(512, 32000)
embed = nn.Embedding(32000, 512)

crit = nn.CrossEntropyLoss()

tokenizer = get_tokenizer("tr")

with open("data/tr/tur.train.src", encoding="utf-8") as infile:
    src_raw = [torch.LongTensor([1] + tokenizer.EncodeAsIds(infile.readline().strip()) + [2]) for x in range(10)]

with open("data/tr/tur.train.tgt", encoding="utf-8") as infile:
    tgt_raw = [torch.LongTensor([1] + tokenizer.EncodeAsIds(infile.readline().strip()) + [2]) for x in range(10)]


from reformer_pytorch import ReformerEncDec

DE_SEQ_LEN = 1024
EN_SEQ_LEN = 1024

enc_dec = ReformerEncDec(
    dim=512,
    enc_num_tokens=32000,
    enc_depth=6,
    enc_max_seq_len=DE_SEQ_LEN,
    dec_num_tokens=32000,
    dec_depth=6,
    dec_max_seq_len=EN_SEQ_LEN,
    ignore_index=3,
    pad_value=3
).cuda()

train_seq_in = torch.transpose(nn.utils.rnn.pad_sequence(src_raw, padding_value=3), 0, 1).cuda() # 10, 117
train_seq_out = torch.transpose(nn.utils.rnn.pad_sequence(tgt_raw, padding_value=3), 0, 1).cuda() # 10 439
print(train_seq_in.shape, train_seq_out.shape)
input_mask = torch.ones(train_seq_in.shape[0], train_seq_in.shape[1]).bool().cuda()
print(input_mask.shape)

for i in range(1000):
    loss = enc_dec(train_seq_in, train_seq_out, return_loss = True, enc_input_mask = input_mask)
    print(loss)
    loss.backward()
# learn
#
# # evaluate with the following
# eval_seq_in = torch.randint(0, 20000, (1, DE_SEQ_LEN)).long().cuda()
# eval_seq_out_start = torch.tensor([[0.]]).long().cuda() # assume 0 is id of start token
# samples = enc_dec.generate(eval_seq_in, eval_seq_out_start, seq_len = EN_SEQ_LEN, eos_token = 1) # assume 1 is id of stop token
# print(samples.shape) # (1, <= 1024) decode the tokens
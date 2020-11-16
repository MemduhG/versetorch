# -*- coding: utf-8 -*-
from batch import rebatch
from data import get_dataset, get_training_iterators
from loss_optim import MultiGPULossCompute, SimpleLossCompute
from model import make_model, NoamOpt, LabelSmoothing, translate_sentence
from utils import get_tokenizer

import time
import argparse
import torch
from torch import nn


def run_epoch(data_iter, model, loss_compute, tokenizer):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    def sanity_check():
        if model.module is not None:
            mod = model.module
        else:
            mod = model
        translate_sentence(mod, sent="Hak yoluna gidenleriz.", tokenizer=tokenizer)
    sanity_check()
    for i, batch in enumerate(data_iter):
        model.train()
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            sanity_check()
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def run_training(train_iter, valid_iter, tokenizer, epochs=10, vocab_size=32000, config_name=None):
    pad_idx = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(vocab_size, vocab_size, N=6).to(device)
    criterion = LabelSmoothing(size=vocab_size, padding_idx=pad_idx, smoothing=0.1)

    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print("Training with {} GPUs.".format(device_count))
        devices = [x for x in range(device_count)]
        loss_train = MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt)
        loss_val = MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None)
        model_par = nn.DataParallel(model, device_ids=devices)
        for epoch in range(epochs):
            model_par.train()
            run_epoch((rebatch(pad_idx, b) for b in train_iter), model_par, loss_train, tokenizer)
            model_par.eval()
            loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model_par, loss_val, tokenizer)
            print(loss)
    else:
        loss_train = SimpleLossCompute(model.generator, criterion, model_opt)
        loss_val = SimpleLossCompute(model.generator, criterion, None)
        for epoch in range(epochs):
            model.train()
            run_epoch((rebatch(pad_idx, b) for b in train_iter), model, loss_train, tokenizer)
            model.eval()
            loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model, loss_val, tokenizer)
            print(loss)


if __name__ == "__main__":
    train, val, test = get_training_iterators("antoloji")
    tokenizer = get_tokenizer("tr")
    run_training(train, val, tokenizer)

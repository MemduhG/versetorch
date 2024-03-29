#!/storage/praha1/home/memduh/versetorch/venv python
import argparse
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data_utils.batch import rebatch
from src.data_utils.data import get_training_iterators
from src.model.loss_optim import MultiGPULossCompute, SimpleLossCompute
from src.model.model import make_model, NoamOpt, LabelSmoothing, translate_sentence
from src.utils.utils import get_tokenizer
from src.utils.qsub import qsub

from torch.utils.tensorboard import SummaryWriter
import os
import time
import torch
from torch import nn

t = time.time()
last_saved = t

writer = SummaryWriter()

def run_epoch(data_iter, model, loss_compute, tokenizer, save_path=None, 
validate=False, criterion=None, model_opt=None, exp_name=None):
    """Standard Training and Logging Function"""
    global t, last_saved
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    if torch.cuda.device_count() > 1:
        mod = model.module
    else:
        mod = model
    for i, batch in enumerate(data_iter):
        if validate is False:
            model.train()
        if torch.cuda.device_count() == 1:
            try:
                out = model.forward(batch.src.to("cuda"), batch.trg.to("cuda"),
                                 batch.src_mask.to("cuda"), batch.trg_mask.to("cuda"))
            except RuntimeError:
                print("OOM - skipping batch", i)
                print("SRC shape:", batch.src.shape(), "TGT shape:", batch.tgt.shape())
                continue
        else:
            out = model.forward(batch.src, batch.trg,
                                 batch.src_mask, batch.trg_mask)
        if torch.cuda.device_count() == 1:
            loss = loss_compute(out, batch.trg_y.to("cuda"), batch.ntokens)
        else:
            loss = loss_compute(out, batch.trg_y, batch.ntokens)

        writer.add_scalar(exp_name + "/Loss", float(loss) , global_step=model.steps)
        writer.add_scalar(exp_name + "/Learning Rate", loss_compute.opt._rate, global_step=model.steps)
        total_loss += float(loss) 
        del out
        ntokens = batch.ntokens
        total_tokens += ntokens
        tokens += ntokens
        del batch
        if loss_compute is not None:
            mod.steps += 1
            if save_path is not None:
                if (time.time() - last_saved > 1800) or (not os.path.exists(save_path)) \
                        or len(os.listdir(save_path)) == 0:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_file = save_path + "/" + str(mod.steps) + ".pt"
                    torch.save({
                        'model_state_dict': mod.state_dict(),
                        'optimizer_state_dict': loss_compute.opt.optimizer.state_dict()},
                        save_file)
                    last_saved = time.time()
                    qsub(save_file, mod.steps)
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / ntokens, tokens / elapsed))
            # sanity_check()
            writer.flush()
            start = time.time()
            tokens = 0
        del loss
        torch.cuda.empty_cache()
    return total_loss / total_tokens


def run_training(dataset, tokenizer, epochs=1000000, vocab_size=32000, config_name=None):
    bsz = 4000 if dataset == "cz" else 4500
    train_iter, valid_iter, test_iter, train_idx, dev_idx, test_idx = get_training_iterators(dataset, batch_size=bsz)
    if config_name is None:
        config_name = "baseline"
    save_path = "checkpoints/" + dataset + "-" + config_name
    exp_name = dataset + "-" + config_name
    pad_idx = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(vocab_size, vocab_size, N=6).to(device)
    criterion = LabelSmoothing(size=vocab_size, padding_idx=pad_idx, smoothing=0.1)

    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
        last = sorted(os.listdir(save_path), reverse=True, key=lambda x: int(x.partition(".")[0]))[0]
        last_file = os.path.join(save_path, last)
        checkpoint = torch.load(last_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        steps = int(last.split(".")[0])
        model_opt._step = steps
        model.steps = steps

    device_count = torch.cuda.device_count()
    if device_count > 1:
        print("Training with {} GPUs.".format(device_count))
        devices = [x for x in range(device_count)]

        backup_val = SimpleLossCompute(model.generator, criterion, opt=None)
        loss_train = MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt)
        loss_val = MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None)
        model_par = nn.DataParallel(model, device_ids=devices)
        for epoch in range(epochs):
            model_par.train()
            run_epoch((rebatch(pad_idx, b) for b in train_iter), model_par, loss_train, tokenizer, save_path=save_path, criterion=criterion,
                      model_opt=model_opt)
            model_par.eval()
            loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model_par, loss_val, tokenizer)
            print(loss)
    else:
        print("Training with 1 GPU.")
        model = model.to(device)
        loss_train = SimpleLossCompute(model.generator, criterion, model_opt)
        loss_val = SimpleLossCompute(model.generator, criterion, None)
        for epoch in range(epochs):
            model.train()
            run_epoch((rebatch(pad_idx, b) for b in train_iter), model, loss_train, 
            tokenizer, save_path=save_path, exp_name=exp_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tur")
    args = parser.parse_args()
    dataset_lang = {"tur": "tr", "eng": "en", "cz": "cz"}
    tokenizer = get_tokenizer(dataset_lang[args.dataset])
    run_training(args.dataset, tokenizer)

